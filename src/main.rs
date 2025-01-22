mod bbox;
mod request_handler;
mod structs;

use anyhow::Result;
use clap::Parser;
use hyper::{server::conn::http1, service::service_fn};
use hyper_util::rt::TokioIo;
use request_handler::handle_request;
use rusqlite::{Connection, OpenFlags};
use std::{
    io::{stdout, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};
use structs::SourceWithLimits;
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Address to listen on. Default 127.0.0.1:3003
    #[arg(short, long, default_value_t = SocketAddr::from(([127, 0, 0, 1], 3003)))]
    listen_address: SocketAddr,

    /// Raster file
    #[arg(short, long, required = true)]
    source: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    print!("Querying boundaries ...");

    stdout().flush()?;

    let sources = args
        .source
        .into_iter()
        .map(|source| {
            let conn = Connection::open_with_flags(&source, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

            conn.query_row(
                concat!(
                    "SELECT ",
                    "MIN(zoom_level), ",
                    "MAX(zoom_level), ",
                    "MIN(tile_column), ",
                    "MAX(tile_column), ",
                    "MIN(tile_row), ",
                    "MAX(tile_row) ",
                    "FROM tiles"
                ),
                (),
                |row| {
                    Ok(SourceWithLimits {
                        source,
                        min_zoom: row.get::<_, u32>(0)?,
                        max_zoom: row.get::<_, u32>(1)?,
                        min_x: row.get::<_, u32>(2)?,
                        max_x: row.get::<_, u32>(3)?,
                        min_y: row.get::<_, u32>(4)?,
                        max_y: row.get::<_, u32>(5)?,
                    })
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!(" done.");

    // Create a dedicated Tokio runtime for Dataset tasks.
    let dataset_runtime = Arc::new(
        tokio::runtime::Builder::new_current_thread()
            // .worker_threads(1)
            .max_blocking_threads(thread::available_parallelism()?.into())
            .enable_all()
            .on_thread_stop(|| {
                println!("thread stopping");
            })
            .on_thread_start(|| {
                println!("thread starting");
            })
            .build()?,
    );

    let sources: &'static [SourceWithLimits<&Path>] = Box::leak(
        Box::leak(sources.into_boxed_slice())
            .iter()
            .map(|source| SourceWithLimits {
                source: source.source.as_path(),
                min_zoom: source.min_zoom,
                max_zoom: source.max_zoom,
                min_x: source.min_x,
                max_x: source.max_x,
                min_y: source.min_y,
                max_y: source.max_y,
            })
            .collect::<Vec<SourceWithLimits<&Path>>>()
            .into_boxed_slice(),
    );

    let listener = TcpListener::bind(args.listen_address).await?;

    println!("Listening on {}", args.listen_address);

    loop {
        let (stream, _) = listener.accept().await?;

        let io = TokioIo::new(stream);

        let pool = dataset_runtime.clone();

        let sfn = service_fn(move |req| {
            let pool = pool.clone();

            async move { handle_request(pool, req, sources).await }
        });

        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new().serve_connection(io, sfn).await {
                eprintln!("Error serving connection: {err:?}");
            }
        });
    }
}
