//! Integration tests that exercise `ImmichClient` end-to-end against a
//! tiny in-process HTTP mock built on `std::net::TcpListener`. These
//! verify the real network code path (URL routing, headers, JSON
//! encoding, status handling) without depending on a live Immich.

use immich_cli::client::{ImmichClient, SearchBackend};
use immich_cli::models::SearchRequest;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
struct CapturedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl CapturedRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

struct MockServer {
    base_url: String,
    captured: std::sync::Arc<std::sync::Mutex<Vec<CapturedRequest>>>,
    _shutdown: Sender<()>,
}

impl MockServer {
    /// `responder` is invoked for every request and returns
    /// `(status_code, status_text, body_json)`.
    fn start<F>(responder: F) -> Self
    where
        F: Fn(&CapturedRequest) -> (u16, &'static str, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost");
        listener
            .set_nonblocking(false)
            .expect("blocking listener");
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_thread = captured.clone();
        let (tx, rx) = mpsc::channel::<()>();

        thread::spawn(move || {
            listener
                .set_nonblocking(true)
                .expect("nonblocking listener");
            loop {
                if rx.try_recv().is_ok() {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .ok();
                        let req = match read_request(&mut stream) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        captured_for_thread.lock().unwrap().push(req.clone());
                        let (status, status_text, body) = responder(&req);
                        let resp = format!(
                            "HTTP/1.1 {status} {status_text}\r\n\
                             Content-Type: application/json\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\
                             \r\n{}",
                            body.len(),
                            body,
                        );
                        let _ = stream.write_all(resp.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            base_url,
            captured,
            _shutdown: tx,
        }
    }

    fn captured(&self) -> Vec<CapturedRequest> {
        self.captured.lock().unwrap().clone()
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> std::io::Result<CapturedRequest> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    let method = parts.first().copied().unwrap_or("").to_string();
    let path = parts.get(1).copied().unwrap_or("").to_string();

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(CapturedRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn ok_body(items: &[&str], next_page: Option<&str>) -> String {
    let items_json: Vec<serde_json::Value> = items
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": format!("id-{}", p.len()),
                "originalPath": p,
                "originalFileName": p.rsplit('/').next().unwrap_or(p),
                "type": "IMAGE",
                "fileCreatedAt": "2025-01-01T00:00:00Z",
                "localDateTime": "2025-01-01T00:00:00Z",
            })
        })
        .collect();
    let next_value: serde_json::Value = match next_page {
        Some(s) => serde_json::Value::String(s.into()),
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "albums": {"total": 0, "count": 0, "items": [], "facets": []},
        "assets": {
            "total": items.len() as u32,
            "count": items.len() as u32,
            "items": items_json,
            "facets": [],
            "nextPage": next_value,
        }
    })
    .to_string()
}

#[test]
fn smart_query_hits_smart_endpoint_with_api_key() {
    let server = MockServer::start(|_| (200, "OK", ok_body(&["/mnt/x/a.jpg"], None)));
    let client = ImmichClient::with_base_url(server.base_url.clone(), "secret-token", 5).unwrap();
    let req = SearchRequest {
        query: Some("beach".into()),
        ..Default::default()
    };
    let resp = client.search(&req).unwrap();
    assert_eq!(resp.assets.items.len(), 1);

    let captured = server.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].method, "POST");
    assert_eq!(captured[0].path, "/api/search/smart");
    assert_eq!(captured[0].header("x-api-key"), Some("secret-token"));
    let body: serde_json::Value = serde_json::from_str(&captured[0].body).unwrap();
    assert_eq!(body["query"], "beach");
}

#[test]
fn metadata_search_hits_metadata_endpoint() {
    let server = MockServer::start(|_| (200, "OK", ok_body(&[], None)));
    let client = ImmichClient::with_base_url(server.base_url.clone(), "k", 5).unwrap();
    let req = SearchRequest {
        country: Some("China".into()),
        ..Default::default()
    };
    client.search(&req).unwrap();
    let captured = server.captured();
    assert_eq!(captured[0].path, "/api/search/metadata");
    let body: serde_json::Value = serde_json::from_str(&captured[0].body).unwrap();
    assert_eq!(body["country"], "China");
    assert!(body.get("query").is_none(), "query should be omitted");
}

#[test]
fn non_2xx_status_surfaces_body_in_error() {
    let server = MockServer::start(|_| (401, "Unauthorized", "{\"message\":\"bad token\"}".into()));
    let client = ImmichClient::with_base_url(server.base_url.clone(), "k", 5).unwrap();
    let err = client
        .search(&SearchRequest {
            query: Some("x".into()),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("401"), "got: {err}");
    assert!(err.contains("bad token"), "got: {err}");
}

#[test]
fn pagination_response_decodes_string_next_page() {
    let server = MockServer::start(|_| (200, "OK", ok_body(&["/mnt/x/a.jpg"], Some("2"))));
    let client = ImmichClient::with_base_url(server.base_url.clone(), "k", 5).unwrap();
    let resp = client
        .search(&SearchRequest {
            query: Some("x".into()),
            ..Default::default()
        })
        .unwrap();
    // The bug we fixed: nextPage must decode to Some(...) (not None) so the
    // search loop knows to ask for page 2. The exact wrapper type is JSON
    // Value; what matters is that it isn't null.
    let np = resp.assets.next_page.expect("nextPage should be present");
    assert!(!np.is_null());
    assert_eq!(np.as_str(), Some("2"));
}

#[test]
fn null_next_page_decodes_to_none_or_null() {
    let server = MockServer::start(|_| (200, "OK", ok_body(&["/mnt/x/a.jpg"], None)));
    let client = ImmichClient::with_base_url(server.base_url.clone(), "k", 5).unwrap();
    let resp = client
        .search(&SearchRequest {
            query: Some("x".into()),
            ..Default::default()
        })
        .unwrap();
    // Either field absent (None) or JSON null both mean "no more pages",
    // and both must be treated the same by callers.
    let no_more = resp
        .assets
        .next_page
        .as_ref()
        .map(|v| v.is_null())
        .unwrap_or(true);
    assert!(no_more);
}
