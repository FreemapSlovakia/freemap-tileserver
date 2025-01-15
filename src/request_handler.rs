use either::Either;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::{
    body::{Bytes, Incoming},
    Method, Request, Response, StatusCode,
};
use image::{codecs::jpeg::JpegDecoder, ImageBuffer, ImageDecoder, ImageError, Rgba, RgbaImage};
use rusqlite::{Connection, ToSql};
use std::convert::Infallible;
use std::{cell::RefCell, sync::Arc};
use std::{io::Cursor, path::Path};
use tokio::runtime::Runtime;
use tokio::task::JoinError;
use url::Url;

thread_local! {
    static THREAD_LOCAL_DATA: RefCell<Vec<(Box<str>, Connection)>> = const {RefCell::new(Vec::new())};
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

    let parts: Vec<_> = url
        .path()
        .get(1..)
        .unwrap_or_default()
        .splitn(3, '/')
        .map(|a| a.parse::<u32>().ok())
        .collect();

    fn get_blobs<'a, T: ToSql>(
        source: &str,
        data: &'a mut Vec<(Box<str>, Connection)>,
        zoom: T,
        x: T,
        y: T,
    ) -> rusqlite::Result<Option<(Vec<u8>, Vec<u8>)>> {
        let conn = if let Some(index) = data.iter().position(|a| a.0.as_ref() == source) {
            &data[index].1
        } else {
            data.push((source.into(), Connection::open(source)?));
            &data.last().unwrap().1
        };

        let mut stmt = conn.prepare(concat!(
            "SELECT tile_data, tile_alpha ",
            "FROM tiles ",
            "WHERE zoom_level = ?1 AND tile_column = ?2 AND tile_row = ?3"
        ))?;

        let mut rows = stmt.query([zoom, x, y])?;

        let Some(row) = rows.next()? else {
            return Ok(None);
        };

        let tile_data = row.get::<_, Vec<u8>>(0)?;

        let tile_alpha = row.get::<_, Vec<u8>>(1)?;

        return Ok(Some((tile_data, tile_alpha)));
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

                    let r1 = loop {
                        let Some(source) = iter.next() else {
                            break None;
                        };

                        let Some((tile_data, tile_alpha)) = get_blobs(source, data, zoom, x, y)?
                        else {
                            continue;
                        };

                        break Some((tile_data, tile_alpha));
                    };

                    let Some((tile_data, tile_alpha)) = r1 else {
                        return Err(ProcessingError::HttpError(StatusCode::NOT_FOUND, None));
                    };

                    if tile_alpha.is_empty() {
                        return Ok(Bytes::from(tile_data));
                    }

                    let mut either: Either<Vec<u8>, ImageBuffer<Rgba<u8>, Vec<u8>>> =
                        Either::Left(tile_data);

                    while let Some(source) = iter.next() {
                        let Some((tile_data2, tile_alpha2)) = get_blobs(source, data, zoom, x, y)?
                        else {
                            continue;
                        };

                        match &either {
                            Either::Left(tile_data) => {
                                let cursor = Cursor::new(tile_data);
                                let decoder = JpegDecoder::new(cursor)?;

                                let (w, h) = decoder.dimensions();

                                let mut tile_data = vec![0; decoder.total_bytes() as usize];
                                decoder.read_image(&mut tile_data)?;

                                let mut rgba_img = RgbaImage::new(w, h);

                                for (i, pixel) in rgba_img.pixels_mut().enumerate() {
                                    let rgb_index = i * 3;

                                    *pixel = Rgba([
                                        tile_data[rgb_index],     // R
                                        tile_data[rgb_index + 1], // G
                                        tile_data[rgb_index + 2], // B
                                        tile_alpha[i],            // A
                                    ]);
                                }

                                either = Either::Right(rgba_img);
                            }
                            Either::Right(_) => todo!("compose"),
                        }
                    }

                    match either {
                        Either::Left(tile_data) => Ok(Bytes::from(tile_data)),
                        Either::Right(image_buffer) => {
                            let mut out = vec![];

                            jpeg_encoder::Encoder::new(&mut out, 90 /* TODO cfg */).encode(
                                &image_buffer,
                                image_buffer.width() as u16,
                                image_buffer.height() as u16,
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
