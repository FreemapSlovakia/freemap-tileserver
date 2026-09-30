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
- `alpha` - Return a partially covered tile as WebP with alpha instead of JPEG blended onto the background. Fully covered tiles stay JPEG, empty ones 404.

`/coverage.bin` returns the coverage of all sources combined, gzipped if the client accepts it. It is computed in the background after startup, answering 503 until then; reading it from large sources takes minutes, so pass `--coverage-cache <file>` to keep it across restarts. The cache is recomputed when a source's path, size or modification time changes. Format: bytes `[1 (version), maxZoom]`, then a quadtree from the z0 tile, depth first, 2 bits per node packed most significant first: `00` no data, `01` full, `10` partial followed by its children `(2x, 2y)`, `(2x+1, 2y)`, `(2x, 2y+1)`, `(2x+1, 2y+1)` unless at `maxZoom` (`--coverage-zoom`, default 14), `11` partial with nothing known below, so all its descendants count as partial. XYZ scheme. A full tile counts as partial unless all its neighbours are full, since the mask edge sharpens at higher zooms. Clients read a full tile as full at every deeper zoom, so a source's tile is called full only if the source is served down to `--coverage-full-to` (default 20): its deepest zoom, plus 8 where the handler upscales it (it has limits). A source stopping short of `maxZoom` is described at its own deepest zoom, as full or `11`; where finer sources have data under an `11` tile, their tree is kept and only its empty parts are written `11`.

## Example

```sh
cargo run --release -- -s /media/martin/14TB/CZ-ORTOFOTO/mbtiles/vychod-v6-merged.mbtiles -s /home/martin/OSM/stred-with-mask.mbtiles -s /home/martin/OSM/vychod-with-mask.mbtiles -s /home/martin/OSM/zapad-w-a.mbtiles
```
