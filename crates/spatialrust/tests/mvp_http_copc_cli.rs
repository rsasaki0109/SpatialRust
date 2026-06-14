#[cfg(all(feature = "mvp", feature = "io-copc-http"))]
mod http_file_server {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    pub struct LocalHttpFileServer {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl LocalHttpFileServer {
        pub fn start(payload: Vec<u8>) -> (Self, String) {
            let payload = Arc::new(payload);
            let stop = Arc::new(AtomicBool::new(false));
            let stop_server = Arc::clone(&stop);
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind local HTTP server");
            listener.set_nonblocking(true).expect("nonblocking listener");
            let addr = listener.local_addr().expect("local addr");
            let url = format!("http://{addr}/scan.copc.laz");

            let handle = thread::spawn(move || {
                while !stop_server.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let data = Arc::clone(&payload);
                            thread::spawn(move || serve_http_file_request(stream, &data));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(5)),
                    }
                }
            });

            (Self { stop, handle: Some(handle) }, url)
        }
    }

    impl Drop for LocalHttpFileServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn serve_http_file_request(mut stream: TcpStream, payload: &[u8]) {
        let mut buffer = [0_u8; 4096];
        let read = match stream.read(&mut buffer) {
            Ok(0) | Err(_) => return,
            Ok(read) => read,
        };
        let request = String::from_utf8_lossy(&buffer[..read]);
        let request_line = request.lines().next().unwrap_or_default();

        if request_line.starts_with("HEAD ") {
            write_response(&mut stream, 200, payload, None);
            return;
        }

        if !request_line.starts_with("GET ") {
            return;
        }

        let range = request.lines().find_map(|line| line.strip_prefix("Range: bytes="));
        if let Some(range) = range {
            let (start, end) =
                parse_byte_range(range).unwrap_or((0, payload.len().saturating_sub(1)));
            write_response(&mut stream, 206, payload, Some((start, end)));
            return;
        }

        write_response(&mut stream, 200, payload, Some((0, payload.len().saturating_sub(1))));
    }

    fn parse_byte_range(value: &str) -> Option<(usize, usize)> {
        let (start, end) = value.split_once('-')?;
        let start = start.parse().ok()?;
        let end = end.parse().ok()?;
        Some((start, end))
    }

    fn write_response(
        stream: &mut TcpStream,
        status: u16,
        payload: &[u8],
        range: Option<(usize, usize)>,
    ) {
        let (body, content_range) = match range {
            Some((start, end)) if start < payload.len() => {
                let end = end.min(payload.len().saturating_sub(1));
                let slice = &payload[start..=end];
                (slice.to_vec(), Some(format!("bytes {start}-{end}/{}", payload.len())))
            }
            _ => (payload.to_vec(), None),
        };

        let status_text = if status == 206 { "Partial Content" } else { "OK" };
        let mut headers = format!(
            "HTTP/1.1 {status} {status_text}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n",
            body.len()
        );
        if let Some(content_range) = content_range {
            headers.push_str(&format!("Content-Range: {content_range}\r\n"));
        }
        headers.push_str("\r\n");
        let _ = stream.write_all(headers.as_bytes());
        let _ = stream.write_all(&body);
    }
}

#[cfg(all(feature = "mvp", feature = "io-copc-http"))]
#[test]
fn mvp_cli_http_copc_bounds_resolution_matches_local_query() {
    use std::process::Command;

    use spatialrust::{
        read_copc_file_with_query, read_copc_url_info, read_copc_url_with_query, CopcQuery,
    };

    const POINT_COUNT: usize = 50_000;
    let input_path = std::env::temp_dir()
        .join(format!("spatialrust_mvp_http_copc_in_{}.copc.laz", std::process::id()));
    let output_path = std::env::temp_dir()
        .join(format!("spatialrust_mvp_http_copc_out_{}.copc.laz", std::process::id()));

    let cloud = {
        use spatialrust::{
            write_copc_file_with_params, CopcWriterParams, DType, FieldSemantic, PointCloudBuilder,
            PointField, StandardSchemas,
        };
        let schema = StandardSchemas::point_xyzi().with_field(PointField::scalar(
            "classification",
            FieldSemantic::Label,
            DType::U8,
        ));
        let mut builder = PointCloudBuilder::new(schema);
        for index in 0..POINT_COUNT {
            let t = index as f32;
            let x = (t * 0.013).fract() * 80.0;
            let y = ((index % 97) as f32) * 0.41;
            let z = ((index % 53) as f32) * 0.023;
            let intensity = (index % 256) as f32;
            let classification = if z < 0.5 { 2.0 } else { 1.0 };
            builder.push_point([x, y, z, intensity, classification]).unwrap();
        }
        for x in 0..10 {
            for y in 0..10 {
                builder
                    .push_point([90.0 + x as f32 * 0.02, y as f32 * 0.02, 2.5, 200.0, 6.0])
                    .unwrap();
            }
        }
        let cloud = builder.build().unwrap();
        write_copc_file_with_params(
            &input_path,
            &cloud,
            &CopcWriterParams { max_points_per_node: 512, max_depth: 10 },
        )
        .unwrap();
        cloud
    };

    let payload = std::fs::read(&input_path).expect("read copc bytes");
    let (_server, url) = http_file_server::LocalHttpFileServer::start(payload);

    let info = read_copc_url_info(&url).expect("http copc info");
    let roi_bounds = spatialrust::CopcBounds::from_ranges((0.0, 40.0), (0.0, 20.0), (-0.01, 0.5));
    let coarse_resolution = info.spacing * 4.0;
    let query = CopcQuery::with_resolution(roi_bounds, coarse_resolution);
    let expected = read_copc_file_with_query(&input_path, &query)
        .expect("local bounds+resolution query")
        .len();
    let remote =
        read_copc_url_with_query(&url, Some(&query)).expect("http bounds+resolution query").len();
    assert_eq!(remote, expected);
    assert!(expected < cloud.len());

    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
    let output = Command::new(bin)
        .args([
            "--leaf-size",
            "4.0",
            "--voxel-policy",
            "cpu",
            "--bounds",
            "0,0,-0.01,40,20,0.5",
            "--resolution",
            &format!("{coarse_resolution}"),
            &url,
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("run HTTP COPC MVP CLI");
    assert!(
        output.status.success(),
        "HTTP COPC CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let loaded: usize = String::from_utf8_lossy(&output.stderr)
        .lines()
        .find_map(|line| line.strip_prefix("input points: "))
        .and_then(|value| value.trim().parse().ok())
        .expect("CLI stderr should report input points");
    assert_eq!(loaded, expected);

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);
}
