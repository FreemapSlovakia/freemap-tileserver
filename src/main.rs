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
    collections::HashMap,
    io::{stdout, Write},
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    thread,
};
use structs::{SourceLimits, SourceWithLimits};
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Address to listen on
    #[arg(short, long, default_value_t = SocketAddr::from(([127, 0, 0, 1], 3003)))]
    listen_address: SocketAddr,

    /// Source file, can be specified multiple times (order matters)
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

            let limits = conn
                .query_row(
                    "SELECT value FROM metadata WHERE name = 'limits'",
                    (),
                    |row| row.get::<_, String>(0),
                )
                .ok();

            let limits = match limits {
                Some(limits) => serde_json::from_str::<HashMap<u8, SourceLimits>>(&limits).unwrap(), // TODO unwrap
                None => {
                    // compute limits

                    let mut stmt = conn.prepare(concat!(
                        "SELECT ",
                        "zoom_level, ",
                        "MIN(tile_column), ",
                        "MAX(tile_column), ",
                        "MIN(tile_row), ",
                        "MAX(tile_row) ",
                        "FROM tiles GROUP BY zoom_level"
                    ))?;

                    let mut rows = stmt.query(())?;

                    let mut source_limits = HashMap::new();

                    while let Some(row) = rows.next()? {
                        source_limits.insert(
                            row.get::<_, u8>(0)?,
                            SourceLimits {
                                min_x: row.get::<_, u32>(1)?,
                                max_x: row.get::<_, u32>(2)?,
                                min_y: row.get::<_, u32>(3)?,
                                max_y: row.get::<_, u32>(4)?,
                            },
                        );
                    }

                    source_limits
                }
            };

            Ok(SourceWithLimits { source, limits })
        })
        .collect::<Result<Vec<_>, rusqlite::Error>>()?;

    println!(" done.");

    // println!("{}", serde_json::to_string(&sources)?);

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

    let sources: &'static Vec<_> = Box::leak(Box::new(sources));

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
