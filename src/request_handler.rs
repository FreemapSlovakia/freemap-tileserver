use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::{
    body::{Bytes, Incoming},
    Method, Request, Response, StatusCode,
};
use image::{codecs::jpeg::JpegDecoder, ImageDecoder, ImageError, Rgba, RgbaImage};
use itertools::Itertools;
use rusqlite::Connection;
use std::{borrow::Cow, convert::Infallible};
use std::{cell::RefCell, sync::Arc};
use std::{io::Cursor, path::Path};
use tokio::runtime::Runtime;
use tokio::task::JoinError;
use url::Url;

thread_local! {
    static THREAD_LOCAL_DATA: RefCell<Vec<(Box<str>, Connection)>> = const {RefCell::new(Vec::new())};
}

enum ImageType {
    Jpeg,
    // Png,
    Webp,
}

pub enum Background {
    Alpha,
    Rgb(u8, u8, u8),
}

pub struct BackgroundError();

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
            .map(Iterator::collect::<String>)
            .map(|c| u8::from_str_radix(&c, 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| BackgroundError())
            .map(|rgb| Self::Rgb(rgb[0], rgb[1], rgb[2]))
    }
}

impl TryFrom<&str> for ImageType {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            _ => Err(format!("unsupported extension {value}")),
        }
    }
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

pub async fn handle_request(
    pool: Arc<Runtime>,
    req: Request<Incoming>,
    raster_path: &'static Path,
) -> Result<Response<BoxBody<Bytes, BodyError>>, hyper::http::Error> {
    let sources = vec![
        "/home/martin/OSM/vychod.mbtiles",
        "/home/martin/OSM/stred.mbtiles",
    ];

    if req.method() != Method::GET {
        return http_error(StatusCode::METHOD_NOT_ALLOWED);
    }

    let url = Url::parse(&format!("http://localhost{}", req.uri().to_string())).unwrap();

    let path = url.path();

    let mut quality = 75.0;

    let mut background = Background::Alpha;

    for pair in url.query_pairs() {
        match pair.0.as_ref() {
            "background" | "bg" => {
                background = match pair.1.try_into() {
                    Ok(bg) => bg,
                    Err(_) => return http_error(StatusCode::BAD_REQUEST),
                }
            }
            "quality" | "q" => {
                quality = match pair.1.parse::<f32>() {
                    Ok(quality) => quality,
                    Err(_) => return http_error(StatusCode::BAD_REQUEST),
                }
            }
            _ => {}
        }
    }

    let parts: Vec<_> = path.splitn(2, '.').collect();

    let ext: Result<Option<ImageType>, _> = parts.get(1).map(|&x| x.try_into()).transpose();

    let Ok(ext) = ext else {
        return http_error(StatusCode::NOT_FOUND);
    };

    let parts: Vec<_> = parts
        .get(0)
        .copied()
        .unwrap_or_default()
        .get(1..)
        .unwrap_or_default()
        .splitn(3, '/')
        .map(|a| a.parse::<u32>().ok())
        .collect();

    match (
        parts.get(0).copied().flatten(),
        parts.get(1).copied().flatten(),
        parts.get(2).copied().flatten(),
    ) {
        (Some(zoom), Some(x), Some(y)) if parts.len() == 3 => pool
            .spawn_blocking(move || {
                THREAD_LOCAL_DATA.with_borrow_mut(|data| {
                    for source in sources {
                        let conn = if let Some(conn) = data.iter().find(|a| a.0.as_ref() == source).map(|a| &a.1) { conn }
                            else  {
                                data.push((source.into(), Connection::open(source)?));

                                &data.last().unwrap().1
                            };

                        let mut stmt = conn.prepare("SELECT tile_data, tile_alpha FROM tiles WHERE zoom_level = ?1 AND tile_column = ?2 AND tile_row = ?3")?;

                        let mut rows = stmt.query([zoom, x, y])?;

                        if let Some(row) = rows.next()? {
                            let tile_data = row.get::<_, Vec<u8>>(0)?;

                            let tile_alpha = row.get::<_, Vec<u8>>(1)?;

                            if tile_alpha.is_empty() {
                                return Ok((ImageType::Jpeg, Bytes::from(tile_data)));
                            }

                            let cursor = Cursor::new(tile_data);
                            let decoder = JpegDecoder::new(cursor)?;

                            let (w, h) = decoder.dimensions();

                            let mut tile_data = vec![0; decoder.total_bytes() as usize];
                            decoder.read_image(&mut tile_data)?;


                            let mut rgba_img = RgbaImage::new(w, h);

                            for (i, pixel) in rgba_img.pixels_mut().enumerate() {
                                let rgb_index = i * 3;

                                *pixel = Rgba([
                                    tile_data[rgb_index],      // R
                                    tile_data[rgb_index + 1],  // G
                                    tile_data[rgb_index + 2],  // B
                                    tile_alpha[i],             // A
                                ]);
                            }


                            // image_buffer.
                        }
                    }

                    Err(ProcessingError::HttpError(StatusCode::NOT_FOUND, None))
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
                |message| {
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(
                            "Content-Type",
                            match message.0 {
                                ImageType::Jpeg => "image/jpeg",
                                ImageType::Webp => "image/webp",
                            },
                        )
                        .header("Access-Control-Allow-Origin", "*")
                        .body(Full::new(message.1).map_err(|e| match e {}).boxed())
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
