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

/// Translate a local NFS path back to the Immich server-side path. The
/// inverse of `translate`: matches local prefix, swaps in server prefix.
/// Used by the `info` subcommand to find an asset given its on-disk path.
pub fn reverse_translate(local_path: &str, mappings: &[PathMapEntry]) -> Option<String> {
    for entry in mappings {
        if let Some(rest) = strip_path_prefix(local_path, &entry.local) {
            let mut server = entry.server.clone();
            if !rest.is_empty() {
                if !rest.starts_with('/') {
                    server.push('/');
                }
                server.push_str(rest);
            }
            return Some(server);
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
        let got = translate("/mnt/qnap-photos/Family/2025/img.jpg", &m).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/QNAP-Photos/Family/2025/img.jpg"));
    }

    #[test]
    fn no_partial_segment_match() {
        let m = vec![entry("/mnt/qnap-photos", "/local")];
        assert!(translate("/mnt/qnap-photos-extra/img.jpg", &m).is_none());
    }

    #[test]
    fn first_match_wins() {
        let m = vec![
            entry("/mnt/qnap-photos/Family", "/local/family"),
            entry("/mnt/qnap-photos", "/local/all"),
        ];
        let got = translate("/mnt/qnap-photos/Family/x.jpg", &m).unwrap();
        assert_eq!(got, PathBuf::from("/local/family/x.jpg"));
    }

    #[test]
    fn unmapped_returns_none() {
        let m = vec![entry("/mnt/qnap-photos", "/local")];
        assert!(translate("/elsewhere/x.jpg", &m).is_none());
    }

    #[test]
    fn reverse_simple() {
        let m = vec![entry("/mnt/qnap-photos", "/home/u/QNAP-Photos")];
        let got = reverse_translate("/home/u/QNAP-Photos/Family/2018/x.jpg", &m).unwrap();
        assert_eq!(got, "/mnt/qnap-photos/Family/2018/x.jpg");
    }

    #[test]
    fn reverse_no_partial_segment_match() {
        // Same boundary rule as forward translate: `/home/u/QNAP-Photos` must
        // not match `/home/u/QNAP-Photos-extra/...`.
        let m = vec![entry("/mnt/q", "/home/u/QNAP-Photos")];
        assert!(reverse_translate("/home/u/QNAP-Photos-extra/x.jpg", &m).is_none());
    }

    #[test]
    fn reverse_first_match_wins() {
        let m = vec![
            entry("/mnt/q/family", "/home/u/Photos/family"),
            entry("/mnt/q", "/home/u/Photos"),
        ];
        let got = reverse_translate("/home/u/Photos/family/x.jpg", &m).unwrap();
        assert_eq!(got, "/mnt/q/family/x.jpg");
    }

    #[test]
    fn reverse_unmapped_returns_none() {
        let m = vec![entry("/mnt/q", "/home/u/Photos")];
        assert!(reverse_translate("/somewhere/else/x.jpg", &m).is_none());
    }

    #[test]
    fn reverse_handles_unicode_segments() {
        // Ensure unicode path segments round-trip unchanged.
        let m = vec![entry("/mnt/qnap-photos", "/home/u/QNAP-Photos")];
        let got =
            reverse_translate("/home/u/QNAP-Photos/Family/2018年/IMG_20180908.jpg", &m).unwrap();
        assert_eq!(got, "/mnt/qnap-photos/Family/2018年/IMG_20180908.jpg");
    }
}
