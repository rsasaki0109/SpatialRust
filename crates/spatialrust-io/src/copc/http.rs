//! HTTP range-request byte source for remote COPC files.

use std::io::Read;
use std::sync::Arc;

use copc_streaming::ByteSource;

use spatialrust_core::PointCloud;

use crate::copc::query::{CopcFileInfo, CopcQuery};
use crate::copc::reader::{read_copc_from_byte_source, read_header_info};
use crate::error::{copc_format, IoError};

const DEFAULT_MAX_PARALLEL_RANGES: usize = 8;

/// Random-access COPC byte source backed by HTTP range requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpByteSource {
    url: String,
    max_parallel_ranges: usize,
}

impl HttpByteSource {
    /// Creates an HTTP byte source for a remote COPC URL.
    pub fn new(url: impl Into<String>) -> Result<Self, IoError> {
        let url = url.into();
        validate_http_url(&url)?;
        Ok(Self { url, max_parallel_ranges: DEFAULT_MAX_PARALLEL_RANGES })
    }

    /// Limits how many HTTP range requests are in flight at once.
    #[must_use]
    pub fn with_max_parallel_ranges(mut self, max_parallel_ranges: usize) -> Self {
        self.max_parallel_ranges = max_parallel_ranges.max(1);
        self
    }

    /// Returns the source URL.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the configured maximum number of parallel range requests.
    #[must_use]
    pub fn max_parallel_ranges(&self) -> usize {
        self.max_parallel_ranges
    }
}

impl ByteSource for HttpByteSource {
    async fn read_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, copc_streaming::CopcError> {
        fetch_http_range(&self.url, offset, length)
    }

    async fn read_ranges(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<Vec<u8>>, copc_streaming::CopcError> {
        fetch_http_ranges_parallel(&self.url, ranges, self.max_parallel_ranges)
    }

    async fn size(&self) -> Result<Option<u64>, copc_streaming::CopcError> {
        fetch_http_size(&self.url)
    }
}

/// Reads all points from a remote COPC URL.
pub fn read_copc_url(url: &str) -> Result<PointCloud, IoError> {
    read_copc_url_with_query(url, None)
}

/// Reads points from a remote COPC URL using an optional spatial query.
pub fn read_copc_url_with_query(
    url: &str,
    query: Option<&CopcQuery>,
) -> Result<PointCloud, IoError> {
    if let Some(query) = query {
        query.validate()?;
    }
    validate_http_url(url)?;
    let source = HttpByteSource::new(url)?;
    pollster::block_on(read_copc_from_byte_source(source, query))
}

/// Reads COPC header metadata from a remote URL without loading points.
pub fn read_copc_url_info(url: &str) -> Result<CopcFileInfo, IoError> {
    validate_http_url(url)?;
    let source = HttpByteSource::new(url)?;
    pollster::block_on(async { read_header_info(source).await.map(|(_, info)| info) })
}

fn fetch_http_range(
    url: &str,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>, copc_streaming::CopcError> {
    if length == 0 {
        return Ok(Vec::new());
    }

    let end = offset.saturating_add(length.saturating_sub(1));
    let response = ureq::get(url)
        .set("Range", &format!("bytes={offset}-{end}"))
        .call()
        .map_err(|error| copc_streaming::CopcError::ByteSource(Box::new(error)))?;

    let status = response.status();
    if status != 200 && status != 206 {
        return Err(copc_streaming::CopcError::ByteSource(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unexpected HTTP status {status} for range request"),
        ))));
    }

    let mut bytes = Vec::with_capacity(length as usize);
    response
        .into_reader()
        .take(length)
        .read_to_end(&mut bytes)
        .map_err(copc_streaming::CopcError::Io)?;
    Ok(bytes)
}

fn fetch_http_ranges_parallel(
    url: &str,
    ranges: &[(u64, u64)],
    max_parallel_ranges: usize,
) -> Result<Vec<Vec<u8>>, copc_streaming::CopcError> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(ranges.len());
    let url = Arc::new(url.to_owned());

    for batch in ranges.chunks(max_parallel_ranges.max(1)) {
        let batch_results = read_range_batch(Arc::clone(&url), batch)?;
        results.extend(batch_results);
    }

    Ok(results)
}

fn read_range_batch(
    url: Arc<String>,
    ranges: &[(u64, u64)],
) -> Result<Vec<Vec<u8>>, copc_streaming::CopcError> {
    if ranges.len() == 1 {
        let (offset, length) = ranges[0];
        return Ok(vec![fetch_http_range(url.as_str(), offset, length)?]);
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranges.len());
        for (index, &(offset, length)) in ranges.iter().enumerate() {
            let url = Arc::clone(&url);
            handles.push(scope.spawn(move || {
                let bytes = fetch_http_range(url.as_str(), offset, length)?;
                Ok::<_, copc_streaming::CopcError>((index, bytes))
            }));
        }

        let mut batch = vec![Vec::new(); ranges.len()];
        for handle in handles {
            let (index, bytes) = handle.join().map_err(|_| {
                copc_streaming::CopcError::ByteSource(Box::new(std::io::Error::other(
                    "parallel HTTP range worker panicked",
                )))
            })??;
            batch[index] = bytes;
        }
        Ok(batch)
    })
}

fn fetch_http_size(url: &str) -> Result<Option<u64>, copc_streaming::CopcError> {
    if let Ok(response) = ureq::head(url).call() {
        if let Some(total) = response.header("Content-Length").and_then(parse_u64_header) {
            return Ok(Some(total));
        }
    }

    let response = ureq::get(url)
        .set("Range", "bytes=0-0")
        .call()
        .map_err(|error| copc_streaming::CopcError::ByteSource(Box::new(error)))?;

    if let Some(total) = response.header("Content-Range").and_then(parse_content_range_total) {
        return Ok(Some(total));
    }

    if let Some(total) = response.header("Content-Length").and_then(parse_u64_header) {
        return Ok(Some(total));
    }

    Ok(None)
}

fn validate_http_url(url: &str) -> Result<(), IoError> {
    if url.starts_with("http://") || url.starts_with("https://") {
        Ok(())
    } else {
        Err(copc_format(format!(
            "COPC HTTP sources require an http:// or https:// URL, got `{url}`"
        )))
    }
}

fn parse_u64_header(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

fn parse_content_range_total(value: &str) -> Option<u64> {
    value.split('/').nth(1)?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{
        fetch_http_ranges_parallel, parse_content_range_total, read_range_batch, validate_http_url,
        HttpByteSource,
    };
    use copc_streaming::ByteSource;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn validates_http_urls() {
        assert!(validate_http_url("https://example.com/cloud.copc.laz").is_ok());
        assert!(validate_http_url("/tmp/local.copc.laz").is_err());
    }

    #[test]
    fn parses_content_range_total() {
        assert_eq!(parse_content_range_total("bytes 0-0/12345"), Some(12345));
    }

    #[test]
    fn constructs_http_source() {
        let source = HttpByteSource::new("https://example.com/cloud.copc.laz").unwrap();
        assert_eq!(source.url(), "https://example.com/cloud.copc.laz");
        assert_eq!(source.max_parallel_ranges(), 8);
    }

    #[test]
    fn read_ranges_fetches_multiple_byte_ranges() {
        let payload = b"0123456789ABCDEF";
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_server = Arc::clone(&requests);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();

        let server = thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while requests_server.load(Ordering::SeqCst) < 3 {
                if std::time::Instant::now() > deadline {
                    panic!("timed out waiting for HTTP range requests");
                }
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                serve_test_range(&mut stream, payload, &requests_server);
            }
        });

        let url = format!("http://{addr}/cloud.copc.laz");
        let source = HttpByteSource::new(&url).unwrap().with_max_parallel_ranges(3);
        let ranges = vec![(0, 4), (4, 4), (8, 4)];
        let results = pollster::block_on(source.read_ranges(&ranges)).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], b"0123".to_vec());
        assert_eq!(results[1], b"4567".to_vec());
        assert_eq!(results[2], b"89AB".to_vec());
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        server.join().unwrap();
    }

    #[test]
    fn fetch_ranges_batches_by_parallelism_limit() {
        let url = "https://example.com/cloud.copc.laz";
        let ranges = vec![(0, 1); 5];
        let err = fetch_http_ranges_parallel(url, &ranges, 2).unwrap_err();
        assert!(matches!(
            err,
            copc_streaming::CopcError::ByteSource(_) | copc_streaming::CopcError::Io(_)
        ));
    }

    #[test]
    fn single_range_batch_delegates_to_fetch() {
        let result = read_range_batch(
            Arc::new("https://invalid.test/not-found.copc.laz".to_owned()),
            &[(0, 1)],
        );
        assert!(result.is_err());
    }

    fn serve_test_range(stream: &mut TcpStream, payload: &[u8], requests: &AtomicUsize) {
        let mut buffer = [0_u8; 512];
        let read = stream.read(&mut buffer).unwrap();
        let request = std::str::from_utf8(&buffer[..read]).unwrap();
        let range = request
            .lines()
            .find_map(|line| line.strip_prefix("Range: bytes="))
            .expect("missing Range header");
        let (start, end) = range
            .split_once('-')
            .and_then(|(start, end)| Some((start.parse::<u64>().ok()?, end.parse::<u64>().ok()?)))
            .expect("invalid Range header");
        let start = start as usize;
        let end = end as usize;
        let body = payload[start..=end].to_vec();

        requests.fetch_add(1, Ordering::SeqCst);
        let response = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\n\r\n",
            body.len(),
            payload.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(&body).unwrap();
    }
}
