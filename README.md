# Humedales Altoandinos - Sentinel-2 y Dron

Proyecto para monitoreo histórico de humedales altoandinos con dos fuentes separadas:

- `Sentinel-2`: series estacionales históricas desde `2017` hasta el año actual.
- `Dron Multiespectral`: estructura activa desde ya, incluso si todavía no tiene campañas cargadas.

El backend genera manifiestos, series temporales, métricas históricas y rasters de visualización web por fuente. El frontend consume esos productos y permite cambiar entre `Sentinel-2` y `Dron` sin mezclar contextos.

## Humedales configurados

- `sapunta`
- `vegas_calama`

El AOI maestro se publica en `data/shapefiles/aoi.geojson` y además se regenera automáticamente a partir de los shapefiles fuente declarados en `config.yaml`.

## Estructura actual

```text
wetland_ortho_monitoring/
├── config.yaml
├── pipeline_utils.py
├── download_sentinel2_gee.py
├── export_data_ortho.py
├── index.html
├── data/
│   ├── shapefiles/
│   │   ├── aoi.geojson
│   │   ├── sapunta/AE_SAP.shp
│   │   └── vegas_calama/vegas_calama.shp
│   ├── sentinel2/
│   │   └── {wetland_id}/{ndvi|ndwi}/{year}_{season}.tif
│   └── drone/
│       └── {wetland_id}/rgb/{year}_{season}.tif
└── data_static/
    ├── sources_manifest.json
    ├── wetlands_aoi.geojson
    ├── sentinel2/
    │   ├── metadata.json
    │   ├── timeseries.json
    │   ├── csv/
    │   └── rasters/
    └── drone/
        ├── metadata.json
        ├── timeseries.json
        ├── csv/
        └── rasters/
```

## Configuración

`config.yaml` ahora concentra:

- AOI maestro: `data/shapefiles/aoi.geojson`
- humedales `sapunta` y `vegas_calama`
- rango histórico con `year_start: 2017` y `year_end: auto`
- fuentes `sentinel2` y `drone`
- estrategia de visualización raster para web (`WEBP`, `upscale_factor`, preservación de píxel)

## Convención de datos

Entrada principal (estructura por índice):

- `data/sentinel2/{wetland_id}/ndvi/{year}_{season}.tif`
- `data/sentinel2/{wetland_id}/ndwi/{year}_{season}.tif`
- `data/drone/{wetland_id}/rgb/{year}_{season}.tif`

Para migrar datos antiguos: `python migrate_data_folders.py`

Salida estática:

- `data_static/sources_manifest.json`
- `data_static/{source}/timeseries.json`
- `data_static/{source}/metadata.json`
- `data_static/{source}/csv/*.csv`
- `data_static/{source}/rasters/*.{webp|png}`

## Qué exporta el backend

Por cada fuente:

- series por humedal e índice
- raster analítico original referenciado en metadata
- raster de visualización optimizado para web
- promedio histórico
- promedio histórico por estación
- anomalía absoluta y porcentual
- resumen anual
- tendencia lineal interanual

## Uso

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Descarga o incorpora insumos:

```bash
python download_sentinel2_gee.py
```

El descargador escribe en `data/sentinel2` y usa el AOI maestro regenerado desde los shapefiles configurados.

3. Genera productos estáticos para la web:

```bash
python export_data_ortho.py
```

4. Levanta un servidor local:

```bash
python -m http.server 8000
```

Luego abre `http://localhost:8000`.

## Visualización web

La interfaz ahora incluye:

- selector de fuente visible
- modo dron activo con estado vacío controlado
- KPIs por humedal con comparación histórica
- anomalías estacionales para NDVI y NDWI
- vista comparativa entre humedales
- resumen anual con tendencia
- overlay raster web-friendly separado del GeoTIFF analítico

## Regeneración recomendada

Cada vez que cambies AOIs, tif o campañas:

1. Revisa `config.yaml`
2. Ejecuta `python export_data_ortho.py`
3. Recarga la página

Si agregas campañas de dron, deposítalas bajo `data/drone/{wetland_id}` y vuelve a exportar para que la fuente deje de aparecer vacía.
