use crate::config::PathMapEntry;
use std::path::PathBuf;

/// Translate an Immich server-side absolute path to its local NFS counterpart.
///
/// Mappings are tried in order; the first server-prefix match wins. If no
/// mapping matches, returns `None` so callers can decide whether to surface
/// the raw server path or skip the asset.
pub fn translate(server_path: &str, mappings: &[PathMapEntry]) -> Option<PathBuf> {
    for entry in mappings {
        if let Some(rest) = strip_path_prefix(server_path, &entry.server) {
            let mut local = PathBuf::from(&entry.local);
            if !rest.is_empty() {
                local.push(rest.trim_start_matches('/'));
            }
            return Some(local);
        }
    }
    None
}

/// Like `str::strip_prefix`, but only matches at a path-component boundary.
/// `/mnt/qnap-photos` matches `/mnt/qnap-photos/foo` but not `/mnt/qnap-photos-extra/foo`.
fn strip_path_prefix<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = path.strip_prefix(prefix)?;
    if rest.is_empty() || rest.starts_with('/') {
        Some(rest)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(server: &str, local: &str) -> PathMapEntry {
        PathMapEntry {
            server: server.into(),
            local: local.into(),
        }
    }

    #[test]
    fn simple_translation() {
        let m = vec![entry("/mnt/qnap-photos", "/home/u/QNAP-Photos")];
        let got = translate("/mnt/qnap-photos/PYL/2025/img.jpg", &m).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/QNAP-Photos/PYL/2025/img.jpg"));
    }

    #[test]
    fn no_partial_segment_match() {
        let m = vec![entry("/mnt/qnap-photos", "/local")];
        assert!(translate("/mnt/qnap-photos-extra/img.jpg", &m).is_none());
    }

    #[test]
    fn first_match_wins() {
        let m = vec![
            entry("/mnt/qnap-photos/PYL", "/local/pyl"),
            entry("/mnt/qnap-photos", "/local/all"),
        ];
        let got = translate("/mnt/qnap-photos/PYL/x.jpg", &m).unwrap();
        assert_eq!(got, PathBuf::from("/local/pyl/x.jpg"));
    }

    #[test]
    fn unmapped_returns_none() {
        let m = vec![entry("/mnt/qnap-photos", "/local")];
        assert!(translate("/elsewhere/x.jpg", &m).is_none());
    }
}
