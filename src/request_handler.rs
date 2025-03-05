use crate::{
    background::Background,
    structs::{Context, SourceWithLimits, TileData, TileShift},
};
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Bytes, Incoming},
};
use image::{
    DynamicImage, ImageError, ImageReader, Pixel, RgbImage, RgbaImage, imageops::crop_imm,
};
use rusqlite::{Connection, OpenFlags};
use std::{borrow::Cow, cell::RefCell, convert::Infallible, io::Cursor, path::Path, sync::Arc};
use tokio::{runtime::Runtime, task::JoinError};
use url::Url;

struct SourceConnection<'a> {
    source: &'a Path,
    connection: Connection,
}

thread_local! {
    static SOURCE_CONNECTIONS: RefCell<Vec<SourceConnection>> = const {RefCell::new(Vec::new())};
}

// TODO cfg
const JPEG_QUALITY: u8 = 85;

#[derive(thiserror::Error, Debug)]
enum ProcessingError {
    #[error("join error")]
    JoinError(#[from] JoinError),

    #[error("HTTP error")]
    HttpError(StatusCode, Option<&'static str>),

    #[error("image encoding error: {0}")]
    ImageEncodingError(#[from] ImageError),

    #[error("error reading tile: {0}")]
    ReadError(String),

    #[error("error reading tile: {0}")]
    ReadError2(#[from] rusqlite::Error),

    #[error("jpeg encoding error: {0}")]
    EncodingError(#[from] jpeg_encoder::EncodingError),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<&rusqlite::Error> for ProcessingError {
    fn from(e: &rusqlite::Error) -> Self {
        Self::ReadError(e.to_string())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BodyError {
    #[error("infallible")]
    Infillable(Infallible),
}

enum Image {
    Raw(Vec<u8>),
    Raster(RgbaImage),
}

pub async fn handle_request(
    pool: Arc<Runtime>,
    req: Request<Incoming>,
    context: &'static Context,
) -> Result<Response<BoxBody<Bytes, BodyError>>, hyper::http::Error> {
    if req.method() != Method::GET {
        return http_error(StatusCode::METHOD_NOT_ALLOWED);
    }

    let url = Url::parse(&format!("http://localhost{}", req.uri())).unwrap();

    let tile = url.path().get(1..).unwrap_or_default().replace(".jpg", "");

    let parts: Vec<_> = tile.split('/').collect();

    let tile = (
        parts.get(0).map(|v| v.parse::<u8>().ok()).flatten(),
        parts.get(1).map(|v| v.parse::<u32>().ok()).flatten(),
        parts.get(2).map(|v| v.parse::<u32>().ok()).flatten(),
    );

    let mut background = Cow::Borrowed(&context.default_background);

    let mut fallback_missing = false;

    for pair in url.query_pairs() {
        match pair.0.as_ref() {
            "background" | "bg" => {
                background = match pair.1.parse::<Background>() {
                    Ok(bg) => Cow::Owned(bg),
                    Err(_) => return http_error(StatusCode::BAD_REQUEST),
                }
            }
            "fallback_missing" => {
                fallback_missing = true;
            }
            _ => {}
        }
    }

    match tile {
        (Some(zoom), Some(x), Some(y)) if parts.len() == 3 => pool
            .spawn_blocking(move || {
                let y = (1 << zoom) - 1 - y;

                SOURCE_CONNECTIONS.with_borrow_mut(|source_connections| {
                    let get_upscaled_tile_data =
                        |source_connections: &mut Vec<SourceConnection>,
                         source_with_limits: &'static SourceWithLimits|
                         -> Result<Option<TileData>, rusqlite::Error> {
                            if let Some(tile_data) = get_tile_data(
                                source_with_limits,
                                source_connections,
                                zoom,
                                x,
                                y,
                                context.verbosity,
                            )? {
                                return Ok(Some(tile_data));
                            };

                            let Some(limits) = source_with_limits.limits.as_ref() else {
                                return Ok(None);
                            };

                            let mut zoom = zoom;
                            let mut n = 0;

                            while zoom > 0 && n < 8 && !limits.contains_key(&zoom) {
                                zoom -= 1;
                                n += 1;

                                let tile_data = get_tile_data(
                                    source_with_limits,
                                    source_connections,
                                    zoom,
                                    x >> n,
                                    y >> n,
                                    context.verbosity,
                                )?;

                                if let Some(tile_data) = tile_data {
                                    return Ok(Some(TileData {
                                        shift: Some(TileShift {
                                            x: (x & ((1 << n) - 1)) as u8,
                                            y: (y & ((1 << n) - 1)) as u8,
                                            level: n,
                                        }),
                                        ..tile_data
                                    }));
                                }
                            }

                            return Ok(None);
                        };

                    let mut iter = context.sources.iter();

                    let tile_data = loop {
                        let Some(source_with_limits) = iter.next() else {
                            break None; // no source at all
                        };

                        if let Some(tile_data) =
                            get_upscaled_tile_data(source_connections, source_with_limits)?
                        {
                            break Some(tile_data);
                        }
                    };

                    let Some(tile_data) = tile_data else {
                        return if fallback_missing {
                            let mut out = vec![];

                            let empty: Vec<u8> = background
                                .0
                                .channels()
                                .iter()
                                .take(3)
                                .map(|ch| -> u8 { (*ch).into() })
                                .collect();

                            jpeg_encoder::Encoder::new(&mut out, JPEG_QUALITY).encode(
                                &empty.repeat(256 * 256),
                                256,
                                256,
                                jpeg_encoder::ColorType::Rgb,
                            )?;

                            Ok(Bytes::from(out))
                        } else {
                            Err(ProcessingError::HttpError(StatusCode::NOT_FOUND, None))
                        };
                    };

                    let mut image = if tile_data.alpha.is_empty() && tile_data.shift.is_none() {
                        Image::Raw(tile_data.rgb) // no recompression
                    } else {
                        Image::Raster(to_raster(
                            &tile_data.rgb,
                            decompress_tile_alpha(&tile_data.alpha)?,
                            &tile_data.shift,
                        )?)
                    };

                    for source in iter {
                        let Some(tile_data2) = get_upscaled_tile_data(source_connections, source)?
                        else {
                            continue;
                        };

                        if let Image::Raw(rgb) = image {
                            let raster = to_raster(
                                &rgb,
                                decompress_tile_alpha(&tile_data.alpha)?,
                                &tile_data.shift,
                            )?;

                            image = Image::Raster(raster);
                        }

                        let Image::Raster(ref mut dst) = image else {
                            panic!("reached unreachable");
                        };

                        let mut tile_alpha2 = decompress_tile_alpha(&tile_data2.alpha)?;

                        // this is to remove darkened borders caused by lanczos data+mask resizing
                        let mut to_change = vec![0u8; tile_alpha2.len()];

                        let width = dst.width();

                        let height = dst.height();

                        for x in 0..width {
                            for y in 0..height {
                                if dst.get_pixel(x as u32, y as u32)[3] > 128
                                    && tile_alpha2[x as usize + y as usize * width as usize] < 128
                                {
                                    for ny in -2..=2_i32 {
                                        for nx in -2..=2_i32 {
                                            let val = (nx.abs() + ny.abs()) as u8;

                                            let index = ((y as i32 + ny).clamp(0, 255) * 256
                                                + (x as i32 + nx).clamp(0, 255))
                                                as usize;

                                            to_change[index] = to_change[index].max(match val {
                                                0..=1 => 7,
                                                2 => 1,
                                                3.. => 0,
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        for i in 0..tile_alpha2.len() {
                            tile_alpha2[i] >>= to_change[i];
                        }

                        let src = to_raster(&tile_data2.rgb, tile_alpha2, &tile_data2.shift)?;

                        for (dst, src) in dst.pixels_mut().zip(src.pixels()) {
                            dst.blend(src);
                        }

                        let mut template: Option<RgbaImage> = None;

                        // now remove ugly gaps of touching areas
                        for x in 0..(width as u32) {
                            'main: for y in 0..(height as u32) {
                                if template.as_ref().unwrap_or(dst).get_pixel(x, y)[3] == 255 {
                                    continue;
                                }

                                let mut r = 0;
                                let mut g = 0;
                                let mut b = 0;
                                let mut a = 0;

                                for ny in -2..=2_i32 {
                                    for nx in -2..=2_i32 {
                                        let xx = x as i32 + nx;
                                        let yy = y as i32 + ny;

                                        if xx < 0
                                            || xx > width as i32 - 1
                                            || yy < 0
                                            || yy > height as i32 - 1
                                        {
                                            continue;
                                        }

                                        let px = template
                                            .as_ref()
                                            .unwrap_or(dst)
                                            .get_pixel(xx as u32, yy as u32);

                                        if nx != 0 && ny != 0 && px[3] == 0 {
                                            continue 'main;
                                        }

                                        if nx.abs() < 2 && ny.abs() < 2 {
                                            let alpha = px[3] as u32;

                                            r += px[0] as u32 * alpha;
                                            g += px[1] as u32 * alpha;
                                            b += px[2] as u32 * alpha;
                                            a += alpha;
                                        }
                                    }
                                }

                                if a > 0 {
                                    if template.is_none() {
                                        template.replace(dst.clone());
                                    }

                                    dst.get_pixel_mut(x, y)[0] = (r / a) as u8;
                                    dst.get_pixel_mut(x, y)[1] = (g / a) as u8;
                                    dst.get_pixel_mut(x, y)[2] = (b / a) as u8;
                                    dst.get_pixel_mut(x, y)[3] = 255;
                                }
                            }
                        }
                    }

                    match image {
                        Image::Raw(tile_data) => Ok(Bytes::from(tile_data)),
                        Image::Raster(mut raster) => {
                            for px in raster.pixels_mut() {
                                let mut bg = background.0.clone();

                                bg.blend(px);

                                *px = bg;
                            }

                            let mut out = vec![];

                            jpeg_encoder::Encoder::new(&mut out, JPEG_QUALITY).encode(
                                raster.as_raw(),
                                raster.width() as u16,
                                raster.height() as u16,
                                jpeg_encoder::ColorType::Rgba, // ignores alpha
                            )?;

                            Ok(Bytes::from(out))
                        }
                    }
                })
            })
            .await
            .map_err(ProcessingError::JoinError)
            .and_then(|inner_result| inner_result)
            .map_or_else(
                |e| {
                    if let ProcessingError::HttpError(sc, message) = e {
                        if context.verbosity >= 2 {
                            println!("Responding empty");
                        }

                        http_error_msg(sc, message.unwrap_or_else(|| sc.as_str()))
                    } else {
                        eprintln!("Error: {e}");

                        http_error(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                },
                |data| {
                    if context.verbosity >= 2 {
                        println!("Responding tile");
                    }

                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "image/jpeg")
                        .body(Full::new(data).map_err(|e| match e {}).boxed())
                },
            ),
        _ => http_error(StatusCode::NOT_FOUND),
    }
}

fn http_error(sc: StatusCode) -> Result<Response<BoxBody<Bytes, BodyError>>, hyper::http::Error> {
    http_error_msg(sc, sc.as_str())
}

fn http_error_msg(
    sc: StatusCode,
    message: &str,
) -> Result<Response<BoxBody<Bytes, BodyError>>, hyper::http::Error> {
    Response::builder().status(sc).body(
        Full::new(Bytes::from(message.to_owned()))
            .map_err(BodyError::Infillable)
            .boxed(),
    )
}

fn get_tile_data(
    source_with_limits: &'static SourceWithLimits,
    source_connections: &mut Vec<SourceConnection>,
    zoom: u8,
    x: u32,
    y: u32,
    verbosity: u8,
) -> rusqlite::Result<Option<TileData>> {
    if let Some(limits) = &source_with_limits.limits {
        let Some(limits) = limits.get(&zoom) else {
            return Ok(None);
        };

        if x < limits.min_x || x > limits.max_x || y < limits.min_y || y > limits.max_y {
            return Ok(None);
        }
    }

    let source = source_with_limits.source.as_path();

    let conn = if let Some(index) = source_connections.iter().position(|a| a.source == source) {
        &source_connections[index].connection
    } else {
        source_connections.push(SourceConnection {
            source,
            connection: Connection::open_with_flags(source, OpenFlags::SQLITE_OPEN_READ_ONLY)?,
        });

        &source_connections.last().unwrap().connection
    };

    if verbosity >= 2 {
        println!("Selecting");
    }

    let mut stmt = conn.prepare(concat!(
        "SELECT tile_data, tile_alpha ",
        "FROM tiles ",
        "WHERE zoom_level = ?1 AND tile_column = ?2 AND tile_row = ?3"
    ))?;

    if verbosity >= 2 {
        println!("Querying");
    }

    let mut rows = stmt.query((zoom, x, y))?;

    if verbosity >= 2 {
        println!("Done selecting");
    }

    let tile_data = if let Some(row) = rows.next()? {
        let rgb = row.get::<_, Vec<u8>>(0)?;

        if rgb.len() == 0 {
            None
        } else {
            Some(TileData {
                rgb,
                alpha: row.get::<_, Vec<u8>>(1)?,
                shift: None,
            })
        }
    } else {
        None
    };

    Ok(tile_data)
}

fn decompress_tile_alpha(tile_alpha: &[u8]) -> Result<Vec<u8>, ProcessingError> {
    let tile_alpha = if tile_alpha.is_empty() {
        vec![255; 256 * 256]
    } else {
        zstd::stream::decode_all(tile_alpha)?
    };

    Ok(tile_alpha)
}

fn to_raster(
    tile_data: &[u8],
    alpha: Vec<u8>,
    shift: &Option<TileShift>,
) -> Result<RgbaImage, ProcessingError> {
    let rgb = decode_jpeg(tile_data)?;

    assert_eq!(
        rgb.width() * rgb.height(),
        alpha.len() as u32,
        "Alpha channel length must match the number of pixels"
    );

    let raw: Vec<u8> = rgb
        .pixels()
        .zip(alpha.into_iter())
        .flat_map(|(pixel, a)| [pixel[0], pixel[1], pixel[2], a])
        .collect();

    let rgba_image = RgbaImage::from_raw(rgb.width(), rgb.height(), raw).unwrap();

    let raster = if let Some(shift) = shift {
        scale_and_crop(rgba_image, shift)
    } else {
        rgba_image
    };

    Ok(raster)
}

fn scale_and_crop<'a>(raster: RgbaImage, shift: &TileShift) -> RgbaImage {
    // we are not croppig first for the better quality
    // TODO can create huge images for bigger levels; in that case crop first

    let raster = image::imageops::resize(
        &raster,
        256 << shift.level,
        256 << shift.level,
        image::imageops::FilterType::Lanczos3,
    );

    crop_imm(
        &raster,
        (shift.x as u32) << (9 - shift.level),
        (256_u32 << (shift.level - 1)) - ((shift.y as u32) << (9 - shift.level)),
        256,
        256,
    )
    .to_image()
}

fn decode_jpeg(tile_data: &[u8]) -> Result<RgbImage, ProcessingError> {
    let image =
        ImageReader::with_format(Cursor::new(tile_data), image::ImageFormat::Jpeg).decode()?;

    if let DynamicImage::ImageRgb8(image) = image {
        Ok(image)
    } else {
        Err(ProcessingError::ReadError("not ImageRgb8".into()))
    }
}
