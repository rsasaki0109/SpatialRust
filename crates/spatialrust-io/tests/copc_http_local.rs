// Exercises the HTTP COPC reader; only compiled when the feature is enabled so
// `cargo test --workspace` (default features) does not fail to build it.
#![cfg(feature = "io-copc-http")]

mod common;

#[test]
fn local_http_head_has_no_body() {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let (_server, url) = common::LocalHttpFileServer::start(vec![0xff; 128]);
    let address = url.strip_prefix("http://").unwrap().split('/').next().unwrap();
    let mut stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
    write!(stream, "HEAD /fixture.copc.laz HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let response = String::from_utf8(response).expect("HEAD must not send binary payload bytes");
    assert!(response.contains("Content-Length: 128\r\n"));
    assert!(response.contains("Connection: close\r\n"));
    assert!(response.ends_with("\r\n\r\n"));
}

fn fragmented_request(first: impl FnOnce(&str) -> String, last: &[u8]) -> Vec<u8> {
    fragmented_request_with_server(common::LocalHttpFileServer::start(vec![0xff; 128]), first, last)
}

fn fragmented_request_with_server(
    server: (common::LocalHttpFileServer, String),
    first: impl FnOnce(&str) -> String,
    last: &[u8],
) -> Vec<u8> {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let (_server, url) = server;
    let address = url.strip_prefix("http://").unwrap().split('/').next().unwrap();
    let mut stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(std::time::Duration::from_millis(200))).unwrap();
    stream.write_all(first(address).as_bytes()).unwrap();
    let error = stream.read(&mut [0_u8; 1]).expect_err("response before complete request headers");
    assert!(matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut));
    stream.write_all(last).unwrap();
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    response
}

#[test]
fn local_http_fragmented_head_waits_for_header_terminator() {
    let response = fragmented_request(
        |address| format!("HEAD /fixture.copc.laz HTTP/1.1\r\nHost: {address}\r\n"),
        b"\r\n",
    );
    assert_head_response(response);
}

#[test]
fn local_http_nonblocking_head_waits_for_header_terminator() {
    let response = fragmented_request_with_server(
        common::LocalHttpFileServer::start_with_nonblocking_connections(vec![0xff; 128]),
        |address| format!("HEAD /fixture.copc.laz HTTP/1.1\r\nHost: {address}\r\n"),
        b"\r\n",
    );
    assert_head_response(response);
}

fn assert_head_response(response: Vec<u8>) {
    let response = String::from_utf8(response).expect("HEAD must not send payload bytes");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(response.contains("Content-Length: 128\r\n"));
    assert!(response.ends_with("\r\n\r\n"));
}

#[test]
fn local_http_fragmented_range_waits_for_complete_range() {
    let response = fragmented_request(
        |address| format!("GET /fixture.copc.laz HTTP/1.1\r\nHost: {address}\r\nRange: bytes=4-"),
        b"7\r\n\r\n",
    );
    assert_range_response(response);
}

#[test]
fn local_http_nonblocking_range_waits_for_complete_range() {
    let response = fragmented_request_with_server(
        common::LocalHttpFileServer::start_with_nonblocking_connections(vec![0xff; 128]),
        |address| format!("GET /fixture.copc.laz HTTP/1.1\r\nHost: {address}\r\nRange: bytes=4-"),
        b"7\r\n\r\n",
    );
    assert_range_response(response);
}

fn assert_range_response(response: Vec<u8>) {
    let separator = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let headers = std::str::from_utf8(&response[..separator + 4]).unwrap();
    assert!(headers.starts_with("HTTP/1.1 206 Partial Content\r\n"));
    assert!(headers.contains("Content-Length: 4\r\n"));
    assert!(headers.contains("Content-Range: bytes 4-7/128\r\n"));
    assert_eq!(&response[separator + 4..], &[0xff; 4]);
}

#[test]
fn read_copc_url_with_query_matches_local_file() {
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_io::{
        read_copc_file_with_query, read_copc_url_info, read_copc_url_with_query,
        write_copc_file_with_params, CopcBounds, CopcQuery, CopcWriterParams,
    };

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
    for index in 0..2_000 {
        let x = (index % 64) as f32 * 0.1;
        let y = (index / 64) as f32 * 0.1;
        builder.push_point([x, y, 0.0, (index % 256) as f32]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let path = std::env::temp_dir()
        .join(format!("spatialrust_io_http_copc_{}.copc.laz", std::process::id()));
    write_copc_file_with_params(
        &path,
        &cloud,
        &CopcWriterParams { max_points_per_node: 256, max_depth: 8 },
    )
    .unwrap();
    let payload = std::fs::read(&path).unwrap();
    let (_server, url) = common::LocalHttpFileServer::start(payload);

    let info = read_copc_url_info(&url).expect("http copc info");
    let roi = CopcBounds::from_ranges((0.0, 3.0), (0.0, 3.0), (-0.01, 0.01));
    let resolution = info.spacing * 4.0;
    let query = CopcQuery::with_resolution(roi, resolution);
    let local = read_copc_file_with_query(&path, &query).expect("local query");
    let remote = read_copc_url_with_query(&url, Some(&query)).expect("http query");

    assert_eq!(remote.len(), local.len());
    assert!(remote.len() < cloud.len());

    let _ = std::fs::remove_file(path);
}
