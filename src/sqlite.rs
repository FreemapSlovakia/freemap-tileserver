use rusqlite::{Connection, Error, OpenFlags};
use std::path::Path;

/// Opens an MBTiles file as immutable: no locking and no `-wal`/`-shm`, so reads never write,
/// even to a file left in WAL mode. Sources must not change while served.
pub fn open_immutable(path: &Path) -> rusqlite::Result<Connection> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::InvalidPath(path.to_path_buf()))?;

    let escaped = path_str
        .replace('%', "%25")
        .replace('?', "%3f")
        .replace('#', "%23");

    Connection::open_with_flags(
        format!("file:{escaped}?immutable=1"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
}
