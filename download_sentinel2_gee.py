"""
Descarga composiciones estacionales Sentinel-2 desde Google Earth Engine.
Genera un GeoTIFF por humedal, año, estación e índice dentro de `data/sentinel2`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import requests

from pipeline_utils import ensure_master_aoi, get_source_config, load_config, resolve_years

GEE_PROJECT = "ee-javiermedinam"

SEASON_DATE_RANGES = {
    "verano": ("{y}-01-01", "{y}-03-31"),
    "otono": ("{y}-04-01", "{y}-06-30"),
    "invierno": ("{y}-07-01", "{y}-09-30"),
    "primavera": ("{y}-10-01", "{y}-12-31"),
}


def load_aoi_features(path: str | Path, id_col: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        feature["properties"][id_col]: feature
        for feature in data.get("features", [])
        if id_col in feature.get("properties", {})
    }


def mask_s2_sr(image):
    qa = image.select("QA60")
    scl = image.select("SCL")
    clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    scl_ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(clear).updateMask(scl_ok).divide(10000)


def build_composite(geometry, start: str, end: str, bands: list[str]):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(mask_s2_sr)
    )
    image_count = collection.size().getInfo()
    if image_count == 0:
        return None, 0
    composite = collection.median()
    index_image = composite.normalizedDifference(bands).rename("index").clip(geometry).toFloat()
    return index_image, image_count


def download_image(image, region_geojson: dict, out_path: str | Path, scale: int = 10) -> None:
    url = image.getDownloadURL(
        {
            "scale": scale,
            "crs": "EPSG:4326",
            "region": region_geojson,
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(response.content)


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config()
    years = resolve_years(config)
    source_cfg = get_source_config(config, "sentinel2")
    if not source_cfg.get("download_enabled", True):
        raise RuntimeError("La fuente sentinel2 está deshabilitada para descargas en config.yaml")

    master_aoi_path = ensure_master_aoi(config)
    features = load_aoi_features(master_aoi_path, config["shapefile_id_col"])
    output_root = Path(source_cfg["input_root"])
    indices = config["indices"]

    ee.Initialize(project=GEE_PROJECT)

    print(f"Proyecto GEE : {GEE_PROJECT}")
    print(f"Fuente       : {source_cfg['label']}")
    print(f"Humedales    : {list(features.keys())}")
    print(f"Índices      : {list(indices.keys())}")
    print(f"Años         : {years[0]}-{years[-1]}")
    print(f"Salida       : {output_root.as_posix()}")

    for wetland_id, wetland_cfg in config["wetlands"].items():
        feature = features.get(wetland_id)
        if not feature:
            print(f"\n[WARN] AOI no encontrado para {wetland_id}")
            continue

        geometry = ee.Geometry(feature["geometry"])
        print(f"\n=== {wetland_cfg['name']} ({wetland_id}) ===")

        for year in years:
            for season_key, season_label in config["seasons"].items():
                start_tpl, end_tpl = SEASON_DATE_RANGES[season_key]
                start = start_tpl.format(y=year)
                end = end_tpl.format(y=year)

                for index_name, index_cfg in indices.items():
                    output_path = output_root / wetland_id / f"{year}_{season_key}_{index_name}.tif"
                    if output_path.exists():
                        print(f"  [skip] {output_path.as_posix()}")
                        continue

                    print(
                        f"  {year} {season_label} {index_name.upper()}: {start} -> {end} ... ",
                        end="",
                        flush=True,
                    )
                    image, image_count = build_composite(geometry, start, end, index_cfg["bands"])
                    if image is None:
                        print("sin escenas")
                        continue

                    download_image(image, feature["geometry"], output_path)
                    print(f"{image_count} escenas -> {output_path.as_posix()}")


if __name__ == "__main__":
    main()
