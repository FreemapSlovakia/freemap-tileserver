# Freemap Tileserver

## Data preparation

Compile GDAL for JXL compression support. Use GDAL 3.10 or later.

### Slovakia

Download part of [Ortofotomozaia](https://www.geoportal.sk/sk/zbgis/ortofotomozaika/) SR you want to process from and extract it. Then build VRT:

```sh
gdalbuildvrt -a_srs EPSG:8353 all.vrt *.tif
```

Next create cutline for alpha mask:

1. create a tile index file `gdaltindex tmp.gpkg *.tif && ogr2ogr -f GPKG -t_srs EPSG:8353 index.gpkg tmp.gpkg && rm tmp.gpkg`
1. download `lms_datum_snimkovania_#.zip` where # is 2 or 3
1. dissolve `lms_datum_snimkovania`:
   ```sh
   ogr2ogr \
     -f GPKG \
     sk-area.gpkg \
     /vsizip/lms_datum_snimkovania_2.zip/lms_datum_snimkovania_2_cyklus.shp \
     -nln dissolved \
     -nlt POLYGON \
     -dialect sqlite \
     -sql "SELECT ST_Simplify(ST_MakePolygon(ST_ExteriorRing(ST_Buffer(ST_Union(geometry), 0.00001, 1))), 0.1) AS geometry FROM lms_datum_snimkovania_2_cyklus" \
     -a_srs EPSG:8353
   ```
1. dissolve the tile index
   ```sh
   ogr2ogr \
     -f GPKG \
     zapad-tiles.gpkg \
     index.gpkg \
     -nln tiles \
     -nlt POLYGON\
     -dialect sqlite \
     -sql "SELECT ST_Union(geom) AS geometry FROM 'index'" \
     -a_srs EPSG:8353
   ```
1. create a vector mask
   ```sh
   ogr2ogr -f GPKG combined.gpkg sk-area.gpkg -nln dissolved
   ogr2ogr -f GPKG -update -append combined.gpkg zapad-tiles.gpkg -nln tiles
   ogr2ogr -f GPKG intersection.gpkg combined.gpkg \
     -dialect sqlite \
     -sql "
       SELECT ST_Intersection(a.geometry, b.geometry) AS geometry
       FROM tiles a, dissolved b
       WHERE ST_Intersects(a.geometry, b.geometry)
     " \
     -nln intersection \
     -nlt POLYGON \
     -a_srs EPSG:8353
   ```
1. rasterize the mask
   ```sh
   gdal_rasterize \
     -burn 0 \
     -at -i \
     -init 255 \
     -tap \
     $(gdalinfo -json all.vrt | jq -r '"-te \(.cornerCoordinates.upperLeft[0]) \(.cornerCoordinates.lowerRight[1]) \(.cornerCoordinates.lowerRight[0]) \(.cornerCoordinates.upperLeft[1]) -tr \(.geoTransform[1]) \(-.geoTransform[5])"') \
     -ot Byte \
     -of GTiff \
     -co TILED=YES \
     -co COMPRESS=DEFLATE \
     -co BIGTIFF=YES \
     stred-tilemask.gpkg stred-alpha-mask.tif
   ```

Finally create the warped and masked tif.

```sh
ZOOM_LEVEL=19 # for "2 cyklus" use  19

calc_tr() {
    zoom_level=$1
    echo "scale=20; 2 * 4 * a(1) * 6378137 / (256 * 2 ^ $zoom_level)" | bc -l | sed 's/^\./0./'
}

CT='+proj=pipeline +step +inv +proj=krovak +lat_0=49.5 +lon_0=24.8333333333333 +alpha=30.2881397527778 +k=0.9999 +x_0=0 +y_0=0 +ellps=bessel +step +inv +proj=hgridshift +grids=Slovakia_JTSK03_to_JTSK.gsb +step +proj=krovak +lat_0=49.5 +lon_0=24.8333333333333 +alpha=30.2881397527778 +k=0.9999 +x_0=0 +y_0=0 +ellps=bessel +step +inv +proj=krovak +lat_0=49.5 +lon_0=24.8333333333333 +alpha=30.2881397527778 +k=0.9999 +x_0=0 +y_0=0 +ellps=bessel +step +proj=push +v_3 +step +proj=cart +ellps=bessel +step +proj=helmert +x=485.021 +y=169.465 +z=483.839 +rx=-7.786342 +ry=-4.397554 +rz=-4.102655 +s=0 +convention=coordinate_frame +step +inv +proj=cart +ellps=WGS84 +step +proj=pop +v_3 +step +proj=webmerc +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84'

RES=$(calc_tr $ZOOM_LEVEL)

gdalwarp -s_srs 'EPSG:8353' -t_srs 'EPSG:3857' -ct "$CT" -tr $RES $RES -tap -r lanczos -of GTiff -co TILED=YES -co BIGTIFF=YES -co COMPRESS=JXL -co JXL_DISTANCE=1 -co JXL_LOSSLESS=NO -co NUM_THREADS=ALL_CPUS -wo NUM_THREADS=ALL_CPUS -multi Ortofoto_2021_stred_jtsk_rgb/all.vrt stred-warped-jxl.tif
```

```sh
gdaladdo -r lanczos --config BIGTIFF_OVERVIEW YES --config COMPRESS_OVERVIEW JXL --config JXL_LOSSLESS_OVERVIEW NO --config JXL_DISTANCE_OVERVIEW 1 --config GDAL_NUM_THREADS ALL_CPUS --config NUM_THREADS_OVERVIEW ALL_CPUS -ro stred-warped-jxl.tif
```

### Czech republic

TODO

```sh
gdaltindex vychod-tileindex.gpkg -lyr_name index new/*.jpg

ogr2ogr \
  -f GPKG \
  vychod-tileindex-dissolved.gpkg \
  vychod-tileindex.gpkg \
  -nln dissolved \
  -nlt POLYGON \
  -dialect sqlite \
  -sql "SELECT ST_Union(geom) AS geometry FROM 'index'" \
  -a_srs EPSG:5514

ogr2ogr \
  -f GPKG \
  result.gpkg \
  admin.gpkg \
  -nln tiles \
  -nlt POLYGON \
  -dialect sqlite \
  -sql "SELECT ST_Buffer(geom, 99.5, 16) AS geometry FROM administrative_units" \
  -a_srs EPSG:5514

ogr2ogr -f GPKG -update -append result.gpkg vychod-tileindex-dissolved.gpkg -nln dissolved

ogr2ogr -f GPKG intersection.gpkg result.gpkg \
  -dialect sqlite \
  -sql "
    SELECT ST_Intersection(a.geometry, b.geometry) AS geometry
    FROM tiles a, dissolved b
    WHERE ST_Intersects(a.geometry, b.geometry)
  " \
  -nln intersection \
  -nlt POLYGON \
  -a_srs EPSG:5514

gdalbuildvrt vychod.vrt new/*.jpg

gdal_rasterize \
  -burn 0 \
  -at -i \
  -init 255 \
  -tap \
  $(gdalinfo -json vychod.vrt | jq -r '"-te \(.cornerCoordinates.upperLeft[0]) \(.cornerCoordinates.lowerRight[1]) \(.cornerCoordinates.lowerRight[0]) \(.cornerCoordinates.upperLeft[1]) -tr \(.geoTransform[1]) \(-.geoTransform[5])"') \
  -ot Byte \
  -of GTiff \
  -co TILED=YES \
  -co COMPRESS=DEFLATE \
  -co BIGTIFF=YES \
  intersection.gpkg \
  vychod-alpha-mask.tif



gdal_edit -unsetnodata mask...

nice gdalwarp -s_srs 'EPSG:5514' -t_srs 'EPSG:3857' -tr 0.14929107086948486475 0.14929107086948486475 -tap -r lanczos -of GTiff -co TILED=YES -co BIGTIFF=YES -co COMPRESS=JXL -co JXL_DISTANCE=1 -co JXL_LOSSLESS=NO -co NUM_THREADS=ALL_CPUS -wo NUM_THREADS=ALL_CPUS -multi tif/all.vrt zapad-warped-jxl.tif && nice gdaladdo -r lanczos --config BIGTIFF_OVERVIEW YES --config COMPRESS_OVERVIEW JXL --config JXL_LOSSLESS_OVERVIEW NO --config JXL_DISTANCE_OVERVIEW 1 --config GDAL_NUM_THREADS ALL_CPUS --config NUM_THREADS_OVERVIEW ALL_CPUS -ro zapad-warped-jxl.tif
```
