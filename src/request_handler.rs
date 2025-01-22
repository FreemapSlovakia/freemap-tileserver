use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::{
    body::{Bytes, Incoming},
    Method, Request, Response, StatusCode,
};
use image::{codecs::jpeg::JpegDecoder, ImageDecoder, ImageError};
use itertools::Itertools;
use pix::{
    el::Pixel,
    ops::{DestOver, SrcOver},
    rgb::{Rgba8, Rgba8p},
    Raster,
};
use rusqlite::{Connection, OpenFlags};
use std::{borrow::Cow, convert::Infallible};
use std::{cell::RefCell, sync::Arc};
use std::{io::Cursor, path::Path};
use tokio::runtime::Runtime;
use tokio::task::JoinError;
use url::Url;

use crate::structs::SourceWithLimits;

thread_local! {
    static THREAD_LOCAL_DATA: RefCell<Vec<(&Path, Connection)>> = const {RefCell::new(Vec::new())};
}

#[derive(thiserror::Error, Debug)]
enum ProcessingError {
    #[error("join error")]
    JoinError(#[from] JoinError),

    #[error("not acceptable")]
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
    Raster(Raster<Rgba8p>),
}

struct Background(Rgba8p);

struct BackgroundError();

impl TryFrom<Cow<'_, str>> for Background {
    type Error = BackgroundError;

    fn try_from(value: Cow<'_, str>) -> Result<Self, Self::Error> {
        if value.len() != 6 {
            return Err(BackgroundError());
        }

        value
            .chars()
            .chunks(2)
            .into_iter()
            .map(|chunk| chunk.collect::<String>())
            .map(|c| u8::from_str_radix(&c, 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| BackgroundError())
            .map(|rgb| Self(Rgba8p::new(rgb[0], rgb[1], rgb[2], 255)))
    }
}

pub async fn handle_request(
    pool: Arc<Runtime>,
    req: Request<Incoming>,
    sources: &'static [SourceWithLimits<&Path>],
) -> Result<Response<BoxBody<Bytes, BodyError>>, hyper::http::Error> {
    if req.method() != Method::GET {
        return http_error(StatusCode::METHOD_NOT_ALLOWED);
    }

    let url = Url::parse(&format!("http://localhost{}", req.uri().to_string())).unwrap();

    let parts: Vec<_> = url
        .path()
        .get(1..)
        .unwrap_or_default()
        .splitn(3, '/')
        .map(|a| a.parse::<u32>().ok())
        .collect();

    let mut background = Background(Rgba8p::new(0, 0, 0, 255));

    for pair in url.query_pairs() {
        match pair.0.as_ref() {
            "background" | "bg" => {
                background = match pair.1.try_into() {
                    Ok(bg) => bg,
                    Err(_) => return http_error(StatusCode::BAD_REQUEST),
                }
            }
            _ => {}
        }
    }

    match (
        parts.get(0).copied().flatten(),
        parts.get(1).copied().flatten(),
        parts.get(2).copied().flatten(),
    ) {
        (Some(zoom), Some(x), Some(y)) if parts.len() == 3 => pool
            .spawn_blocking(move || {
                THREAD_LOCAL_DATA.with_borrow_mut(|data| {
                    let mut iter = sources.into_iter();

                    let tile = loop {
                        let Some(source) = iter.next() else {
                            break None;
                        };

                        let Some((tile_data, tile_alpha)) = get_blobs(source, data, zoom, x, y)?
                        else {
                            continue;
                        };

                        break Some((tile_data, tile_alpha));
                    };

                    let Some((tile_data, tile_alpha)) = tile else {
                        // if let Some((tile_data, tile_alpha)) =
                        //     get_blobs(source, data, zoom - 1, x / 2, y / 2)?
                        // {
                        // }

                        let mut out = vec![];

                        let empty: Vec<u8> = background
                            .0
                            .channels()
                            .iter()
                            .take(3)
                            .map(|ch| -> u8 { (*ch).into() })
                            .collect();

                        jpeg_encoder::Encoder::new(&mut out, 90 /* TODO cfg */).encode(
                            &empty.repeat(256 * 256),
                            256,
                            256,
                            jpeg_encoder::ColorType::Rgb,
                        )?;

                        return Ok(Bytes::from(out));

                        // return Err(ProcessingError::HttpError(StatusCode::NOT_FOUND, None));
                    };

                    let mut image = if tile_alpha.is_empty() {
                        Image::Raw(tile_data)
                    } else {
                        Image::Raster(to_raster(&tile_data, decompress_tile_alpha(&tile_alpha)?)?)
                    };

                    while let Some(source) = iter.next() {
                        let Some((tile_data2, tile_alpha2)) = get_blobs(source, data, zoom, x, y)?
                        else {
                            continue;
                        };

                        if let Image::Raw(tile_data) = image {
                            let raster =
                                to_raster(&tile_data, decompress_tile_alpha(&tile_alpha)?)?;

                            image = Image::Raster(raster);
                        }

                        let Image::Raster(ref mut dst) = image else {
                            panic!("reached unreachable");
                        };

                        let mut tile_alpha2 = decompress_tile_alpha(&tile_alpha2)?;

                        // this is to remove darkened borders caused by lanczos data+mask resizing
                        if zoom > 7 {
                            let mut to_change = vec![0u8; tile_alpha2.len()];

                            for x in 0..(256 as usize) {
                                for y in 0..(256 as usize) {
                                    let alpha: u8 = dst.pixel(x as i32, y as i32).alpha().into();

                                    if alpha > 128 && tile_alpha2[x + y * 256] < 128 {
                                        for ny in -2..=2_i32 {
                                            for nx in -2..=2_i32 {
                                                let val = (nx.abs() + ny.abs()) as u8;

                                                let index = ((y as i32 + ny).clamp(0, 255) * 256
                                                    + (x as i32 + nx).clamp(0, 255))
                                                    as usize;

                                                to_change[index] =
                                                    to_change[index].max(match val {
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
                                tile_alpha2[i] = tile_alpha2[i] >> to_change[i];
                            }
                        }

                        let src = to_raster(&tile_data2, tile_alpha2)?;

                        dst.composite_raster((0, 0, 256, 256), &src, (0, 0, 256, 256), SrcOver);
                    }

                    match image {
                        Image::Raw(tile_data) => Ok(Bytes::from(tile_data)),
                        Image::Raster(mut raster) => {
                            let mut out = vec![];

                            raster.composite_color((0, 0, 256, 256), background.0, DestOver);

                            let raster = Raster::<Rgba8>::with_raster(&raster);

                            jpeg_encoder::Encoder::new(&mut out, 90 /* TODO cfg */).encode(
                                raster.as_u8_slice(),
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
                        http_error_msg(sc, message.unwrap_or_else(|| sc.as_str()))
                    } else {
                        eprintln!("Error: {e}");

                        http_error(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                },
                |data| {
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "image/jpeg")
                        .header("Access-Control-Allow-Origin", "*")
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

fn get_blobs<'a>(
    source: &'static SourceWithLimits<&Path>,
    data: &'a mut Vec<(&Path, Connection)>,
    zoom: u32,
    x: u32,
    y: u32,
) -> rusqlite::Result<Option<(Vec<u8>, Vec<u8>)>> {
    if zoom < source.min_zoom
        || zoom > source.max_zoom
        || x < source.min_x
        || x > source.max_x
        || y < source.min_y
        || y > source.max_y
    {
        return Ok(None);
    }

    let source = source.source;

    let conn = if let Some(index) = data.iter().position(|a| a.0 == source) {
        &data[index].1
    } else {
        data.push((
            source,
            Connection::open_with_flags(source, OpenFlags::SQLITE_OPEN_READ_ONLY)?,
        ));
        &data.last().unwrap().1
    };

    let mut stmt = conn.prepare(concat!(
        "SELECT tile_data, tile_alpha ",
        "FROM tiles ",
        "WHERE zoom_level = ?1 AND tile_column = ?2 AND tile_row = ?3"
    ))?;

    let mut rows = stmt.query((zoom, x, y))?;

    let Some(row) = rows.next()? else {
        return Ok(None);
    };

    let tile_data = row.get::<_, Vec<u8>>(0)?;

    let tile_alpha = row.get::<_, Vec<u8>>(1)?;

    return Ok(Some((tile_data, tile_alpha)));
}

fn decompress_tile_alpha(tile_alpha: &[u8]) -> Result<Vec<u8>, ProcessingError> {
    if tile_alpha.is_empty() {
        Ok(vec![255; 256 * 256])
    } else {
        Ok(zstd::stream::decode_all(tile_alpha)?)
    }
}

fn to_raster(tile_data: &[u8], tile_alpha: Vec<u8>) -> Result<Raster<Rgba8p>, ProcessingError> {
    let (w, h, tile_data) = decode_jpeg(tile_data)?;

    let tile: Vec<_> = tile_data
        .chunks_exact(3)
        .zip(tile_alpha)
        .flat_map(|(chunk, v2_item)| chunk.iter().copied().chain(std::iter::once(v2_item)))
        .collect();

    Ok(Raster::<Rgba8p>::with_raster(
        &Raster::<Rgba8>::with_u8_buffer(w, h, tile),
    ))
}

fn decode_jpeg(tile_data: &[u8]) -> Result<(u32, u32, Vec<u8>), ProcessingError> {
    let cursor = Cursor::new(tile_data);

    let decoder = JpegDecoder::new(cursor)?;

    let (w, h) = decoder.dimensions();

    let mut tile_data = vec![0; decoder.total_bytes() as usize];

    decoder.read_image(&mut tile_data)?;

    Ok((w, h, tile_data))
}
