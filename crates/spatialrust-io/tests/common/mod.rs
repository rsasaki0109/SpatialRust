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
        let url = format!("http://{addr}/fixture.copc.laz");

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
        let (start, end) = parse_byte_range(range).unwrap_or((0, payload.len().saturating_sub(1)));
        write_response(&mut stream, 206, payload, Some((start, end)));
        return;
    }

    write_response(&mut stream, 200, payload, Some((0, payload.len().saturating_sub(1))));
}

fn parse_byte_range(value: &str) -> Option<(usize, usize)> {
    let (start, end) = value.split_once('-')?;
    let start = start.parse().ok()?;
    let end = if end.is_empty() { None } else { end.parse().ok() };
    Some((start, end?))
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
