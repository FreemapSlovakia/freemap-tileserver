use flate2::{Compression, write::GzEncoder};
use rusqlite::{Connection, OpenFlags};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

const FORMAT_VERSION: u8 = 1;

/// Zoom levels the tile handler upscales a source by, at most.
pub const MAX_UPSCALE: u8 = 8;

#[derive(Clone, Copy, PartialEq)]
enum Cover {
    Partial,
    Full,
    /// Partial, with nothing known below; written without children.
    Unknown,
}

pub struct CoverageSource {
    pub path: PathBuf,
    /// Whether the tile handler upscales it past its deepest zoom.
    pub upscaled: bool,
}

pub struct Coverage {
    pub raw: Vec<u8>,
    pub gzipped: Vec<u8>,
}

impl Coverage {
    fn new(raw: Vec<u8>) -> anyhow::Result<Self> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());

        encoder.write_all(&raw)?;

        let gzipped = encoder.finish()?;

        println!("Coverage: {} B, gzipped {} B", raw.len(), gzipped.len());

        Ok(Self { raw, gzipped })
    }
}

/// The coverage from `cache` if it was computed from the same sources, else computed and cached.
pub fn load_or_compute(
    sources: &[CoverageSource],
    zoom: u8,
    full_to: u8,
    cache: Option<&Path>,
) -> anyhow::Result<Coverage> {
    let key = cache
        .map(|_| cache_key(sources, zoom, full_to))
        .transpose()?;

    if let (Some(cache), Some(key)) = (cache, &key) {
        match read_cache(cache, key) {
            Ok(Some(raw)) => {
                println!("Coverage loaded from {}", cache.display());

                return Coverage::new(raw);
            }
            Ok(None) => println!("Coverage cache {} is stale or damaged", cache.display()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => eprintln!("Error reading coverage cache: {e}"),
        }
    }

    println!("Computing coverage");

    let raw = compute(sources, zoom, full_to)?;

    if let (Some(cache), Some(key)) = (cache, &key) {
        if let Err(e) = write_cache(cache, key, &raw) {
            eprintln!("Error writing coverage cache: {e}");
        }
    }

    Coverage::new(raw)
}

/// Changes whenever a source is replaced, added, removed or reordered.
fn cache_key(sources: &[CoverageSource], zoom: u8, full_to: u8) -> io::Result<String> {
    let mut key = format!("v{FORMAT_VERSION} z{zoom} full-to {full_to}\n");

    for source in sources {
        let meta = fs::metadata(&source.path)?;

        let mtime = meta
            .modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        key += &format!(
            "{} {} {}{}\n",
            fs::canonicalize(&source.path)?.display(),
            meta.len(),
            mtime.as_nanos(),
            if source.upscaled { "" } else { " no-upscale" }
        );
    }

    Ok(key)
}

/// Cache file: key length (u32 LE), key, coverage. A coverage that doesn't parse whole is rejected.
fn read_cache(cache: &Path, key: &str) -> io::Result<Option<Vec<u8>>> {
    let data = fs::read(cache)?;

    let Some((len, rest)) = data.split_first_chunk::<4>() else {
        return Ok(None);
    };

    let len = u32::from_le_bytes(*len) as usize;

    if rest.get(..len) != Some(key.as_bytes()) {
        return Ok(None);
    }

    let raw = &rest[len..];

    Ok(is_complete(raw).then(|| raw.to_vec()))
}

fn write_cache(cache: &Path, key: &str, raw: &[u8]) -> io::Result<()> {
    let mut data = (key.len() as u32).to_le_bytes().to_vec();

    data.extend_from_slice(key.as_bytes());

    data.extend_from_slice(raw);

    let tmp = cache.with_extension("tmp");

    let mut file = fs::File::create(&tmp)?;

    file.write_all(&data)?;

    file.sync_all()?;

    fs::rename(tmp, cache)
}

/// Whether `raw` is one whole coverage: the header, a tree ending in its last byte, and zero padding.
fn is_complete(raw: &[u8]) -> bool {
    let [FORMAT_VERSION, max_zoom, bits @ ..] = raw else {
        return false;
    };

    let mut pos = 0usize;

    // iterative, so a damaged file can't recurse deeply
    let mut stack = vec![0u8];

    while let Some(z) = stack.pop() {
        let Some(byte) = bits.get(pos / 4) else {
            return false;
        };

        let code = (byte >> (6 - 2 * (pos % 4))) & 3;

        pos += 1;

        if code == 2 && z < *max_zoom {
            stack.extend([z + 1; 4]);
        }
    }

    pos.div_ceil(4) == bits.len()
        && (pos % 4 == 0 || bits[pos / 4] & (0xff >> (2 * (pos % 4))) == 0)
}

/// Coverage of all sources combined, as a quadtree. See README.
fn compute(sources: &[CoverageSource], zoom: u8, full_to: u8) -> anyhow::Result<Vec<u8>> {
    let mut cover = HashMap::<(u32, u32), Cover>::new();

    // Tiles of sources stopping short of `zoom`, kept at their own zoom rather than expanded.
    let mut coarse = vec![HashMap::<(u32, u32), Cover>::new(); zoom as usize];

    for source in sources {
        let conn = Connection::open_with_flags(&source.path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

        let Some(source_max) = conn.query_row("SELECT max(zoom_level) FROM tiles", (), |row| {
            row.get::<_, Option<u8>>(0)
        })?
        else {
            continue;
        };

        let source_zoom = zoom.min(source_max);

        // A full tile is full at every deeper zoom to the client, so only a source served down to
        // `full_to` may call one full; the handler upscales one by `MAX_UPSCALE` at most.
        let served_to = if source.upscaled {
            source_max.saturating_add(MAX_UPSCALE)
        } else {
            source_max
        };

        let covers_below = served_to >= full_to;

        let size = 1u32 << source_zoom;

        let mut stmt = conn.prepare(concat!(
            "SELECT tile_column, tile_row, length(tile_alpha) ",
            "FROM tiles ",
            "WHERE zoom_level = ?1 AND length(tile_data) > 0"
        ))?;

        let mut rows = stmt.query([source_zoom])?;

        while let Some(row) = rows.next()? {
            let (x, row_tms) = (row.get::<_, u32>(0)?, row.get::<_, u32>(1)?);

            if x >= size || row_tms >= size {
                continue;
            }

            let y = size - 1 - row_tms;

            // tiler stores no alpha for fully opaque tiles
            let tile_cover = if row.get::<_, Option<i64>>(2)?.unwrap_or(0) == 0 {
                Cover::Full
            } else {
                Cover::Partial
            };

            // below a coarse tile only a full one says anything: its descendants are full too
            let (map, tile_cover) = if source_zoom == zoom {
                (
                    &mut cover,
                    if covers_below {
                        tile_cover
                    } else {
                        Cover::Partial
                    },
                )
            } else if covers_below && tile_cover == Cover::Full {
                (&mut coarse[source_zoom as usize], Cover::Full)
            } else {
                (&mut coarse[source_zoom as usize], Cover::Unknown)
            };

            let entry = map.entry((x, y)).or_insert(tile_cover);

            if tile_cover == Cover::Full {
                *entry = Cover::Full;
            }
        }
    }

    // levels[z]: covered tiles at z; a parent is full only if all four children are
    let mut levels = vec![HashMap::new(); zoom as usize + 1];

    // coarse unknown tiles, below which anything finer sources have nothing of is unknown too
    let mut unknown_roots = vec![HashSet::<(u32, u32)>::new(); zoom as usize + 1];

    levels[zoom as usize] = surrounded_full(&cover, zoom, Cover::Partial);

    for z in (0..zoom as usize).rev() {
        let mut full_children = HashMap::<(u32, u32), u8>::new();

        for (&(x, y), &c) in &levels[z + 1] {
            *full_children.entry((x >> 1, y >> 1)).or_default() += (c == Cover::Full) as u8;
        }

        levels[z] = full_children
            .into_iter()
            .map(|(k, n)| (k, if n == 4 { Cover::Full } else { Cover::Partial }))
            .collect();

        // a coarse full tile is full whatever finer sources say; a coarse unknown one keeps what
        // they say and only fills their gaps
        for (k, c) in surrounded_full(&coarse[z], z as u8, Cover::Unknown) {
            if c == Cover::Full {
                levels[z].insert(k, Cover::Full);
            } else {
                unknown_roots[z].insert(k);

                levels[z].entry(k).or_insert(Cover::Unknown);
            }
        }
    }

    let mut writer = BitWriter::default();

    write_node(&levels, &unknown_roots, &mut writer, 0, 0, 0, false);

    let mut raw = vec![FORMAT_VERSION, zoom];

    raw.extend(writer.finish());

    let partial_count = levels[zoom as usize]
        .values()
        .filter(|&&c| c == Cover::Partial)
        .count();

    println!(
        "Coverage: {} tiles at zoom {zoom} ({partial_count} partial)",
        levels[zoom as usize].len(),
    );

    Ok(raw)
}

/// `tiles` with a full tile demoted to `demoted` unless surrounded by full ones: its mask edge may
/// sharpen at higher zooms. Columns wrap around the antimeridian; past the poles nothing is lost.
fn surrounded_full(
    tiles: &HashMap<(u32, u32), Cover>,
    zoom: u8,
    demoted: Cover,
) -> HashMap<(u32, u32), Cover> {
    let size = 1i64 << zoom;

    tiles
        .iter()
        .map(|(&(x, y), &c)| {
            let surrounded = c == Cover::Full
                && (-1..=1_i64).all(|dy| {
                    let ny = y as i64 + dy;

                    !(0..size).contains(&ny)
                        || (-1..=1_i64).all(|dx| {
                            let nx = (x as i64 + dx).rem_euclid(size);

                            tiles.get(&(nx as u32, ny as u32)) == Some(&Cover::Full)
                        })
                });

            (
                (x, y),
                if c == Cover::Full && !surrounded {
                    demoted
                } else {
                    c
                },
            )
        })
        .collect()
}

/// Depth first, 2 bits per node: 00 none, 01 full, 10 partial followed by its children in order
/// (2x, 2y), (2x + 1, 2y), (2x, 2y + 1), (2x + 1, 2y + 1), 11 partial with nothing known below. A
/// partial tile at the deepest zoom has no children.
fn write_node(
    levels: &[HashMap<(u32, u32), Cover>],
    unknown_roots: &[HashSet<(u32, u32)>],
    writer: &mut BitWriter,
    z: usize,
    x: u32,
    y: u32,
    under_unknown: bool,
) {
    let under_unknown = under_unknown || unknown_roots[z].contains(&(x, y));

    match levels[z].get(&(x, y)) {
        None if under_unknown => writer.push(0b11),
        None => writer.push(0b00),
        Some(Cover::Full) => writer.push(0b01),
        Some(Cover::Unknown) => writer.push(0b11),
        Some(Cover::Partial) => {
            writer.push(0b10);

            if z + 1 < levels.len() {
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    write_node(
                        levels,
                        unknown_roots,
                        writer,
                        z + 1,
                        2 * x + dx,
                        2 * y + dy,
                        under_unknown,
                    );
                }
            }
        }
    }
}

/// 2-bit codes packed most significant first; the last byte is zero-padded.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    count: usize,
}

impl BitWriter {
    fn push(&mut self, code: u8) {
        if self.count % 4 == 0 {
            self.bytes.push(0);
        }

        *self.bytes.last_mut().unwrap() |= code << (6 - 2 * (self.count % 4));

        self.count += 1;
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Whether `Accept-Encoding` admits gzip. `*` counts only when gzip isn't listed itself.
pub fn accepts_gzip<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    let mut gzip = None;

    let mut any = None;

    for item in values.into_iter().flat_map(|value| value.split(',')) {
        let mut parts = item.split(';').map(str::trim);

        let coding = parts.next().unwrap_or_default();

        let q = parts
            .find_map(|param| param.strip_prefix("q="))
            .map_or(1.0, |q| q.parse::<f32>().unwrap_or(0.0));

        if coding.eq_ignore_ascii_case("gzip") {
            gzip = Some(q);
        } else if coding == "*" {
            any = Some(q);
        }
    }

    gzip.or(any).is_some_and(|q| q > 0.0)
}
