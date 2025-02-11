# Freemap Tileserver

Tileserver for MBTiles with Freemap extensions.

Features:

- seamless blending of multiple raster sources
- overzooming for sources not having tiles of highest zoom levels

## Building and installing

```sh
cargo install --path .
```

## Command options

Use `-h` or `--help` to get description of all available options:

```
Usage: freemap-tileserver [OPTIONS] --source <SOURCE>

Options:
  -l, --listen-address <LISTEN_ADDRESS>
          Address to listen on [default: 127.0.0.1:3003]
  -s, --source <SOURCE>
          Source file, can be specified multiple times (order matters)
  -d, --default-background <DEFAULT_BACKGROUND>
          Default background color [default: ffffff]
  -s, --skip-fallback-bounds-computation
          Skip computing bounds if missing
  -h, --help
          Print help
  -V, --version
          Print version
```

## Sources

Generate sources with [`freemap-tiler`](https://github.com/FreemapSlovakia/freemap-tiler).

## URL

URL uses slippy map schema `/{zoom}/{x}/{y}[.jpg]`.

Query parameters:

- `bg=RRGGBB` - Background color. Default is white.
- `fallback_missing` - Fallback missing tile to empty tile of background color. Default is to return 404.

## Example

```sh
cargo run --release -- -s /media/martin/14TB/CZ-ORTOFOTO/mbtiles/vychod-v6-merged.mbtiles -s /home/martin/OSM/stred-with-mask.mbtiles -s /home/martin/OSM/vychod-with-mask.mbtiles -s /home/martin/OSM/zapad-w-a.mbtiles
```
