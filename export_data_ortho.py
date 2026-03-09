"""
Backend para monitoreo de humedales altoandinos.
Exporta series, manifiestos y previsualizaciones raster por fuente (`sentinel2`, `drone`).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping

from pipeline_utils import (
    ensure_master_aoi,
    get_source_input_roots,
    load_config,
    resolve_years,
)

OUTPUT_DIR = Path("data_static")
SEASON_MID_DATES = {
    "verano": "-02-15",
    "otono": "-05-15",
    "invierno": "-08-15",
    "primavera": "-11-15",
}

COLORMAPS = {
    "RdYlGn": [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 139),
        (255, 255, 191),
        (217, 239, 139),
        (166, 217, 106),
        (102, 189, 99),
        (26, 152, 80),
        (0, 104, 55),
    ],
    "RdYlBu": [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 144),
        (255, 255, 191),
        (224, 243, 248),
        (171, 217, 233),
        (116, 173, 209),
        (69, 117, 180),
        (49, 54, 149),
    ],
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_aoi(path: str | Path, id_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    if id_col not in gdf.columns:
        raise ValueError(f"Columna '{id_col}' no encontrada. Disponibles: {list(gdf.columns)}")
    return gdf


def json_dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def apply_colormap(data: np.ndarray, nodata_mask: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    colors = np.array(COLORMAPS[cmap_name], dtype=np.float64)
    norm = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    idx = norm * (len(colors) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(colors) - 1)
    frac = (idx - lo)[..., np.newaxis]
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., :3] = (colors[lo] * (1 - frac) + colors[hi] * frac).astype(np.uint8)
    rgba[..., 3] = 255
    rgba[nodata_mask, 3] = 0
    return rgba


def build_preview_raster(
    tiff_path: str | Path,
    preview_path: str | Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    visualization_cfg: dict,
) -> dict:
    with rasterio.open(tiff_path) as src:
        data = src.read(1).astype("float32")
        bounds = src.bounds
        native_size = [int(src.width), int(src.height)]
        nodata_value = src.nodata

    if nodata_value is not None:
        nodata_mask = np.isclose(data, nodata_value) | np.isnan(data)
    else:
        nodata_mask = np.isnan(data)
    rgba = apply_colormap(data, nodata_mask, vmin, vmax, cmap_name)

    image = Image.fromarray(rgba, "RGBA")
    upscale_factor = max(1, int(visualization_cfg.get("upscale_factor", 1)))
    if upscale_factor > 1:
        image = image.resize(
            (image.width * upscale_factor, image.height * upscale_factor),
            resample=Image.Resampling.NEAREST,
        )

    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_format = str(visualization_cfg.get("preview_format", "WEBP")).upper()
    save_kwargs = {"lossless": True, "quality": 100} if preview_format == "WEBP" else {"optimize": True}
    image.save(preview_path, format=preview_format, **save_kwargs)

    return {
        "path": preview_path.relative_to(OUTPUT_DIR).as_posix(),
        "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        "format": preview_format,
        "native_size": native_size,
        "display_size": [int(image.width), int(image.height)],
        "render_mode": "pixelated" if visualization_cfg.get("preserve_pixels", True) else "smooth",
        "opacity": float(visualization_cfg.get("opacity", 0.9)),
    }


def extract_zonal_mean(tiff_path: str | Path, geom_wgs84: dict) -> float | None:
    try:
        with rasterio.open(tiff_path) as src:
            geom = geom_wgs84
            if src.crs and str(src.crs) != "EPSG:4326":
                geom = transform_geom("EPSG:4326", src.crs, geom_wgs84)
            out_image, _ = mask(src, [geom], crop=True, nodata=src.nodata)
            array = out_image[0].astype("float32")
            if src.nodata is not None:
                array = np.ma.masked_equal(array, src.nodata)
            else:
                array = np.ma.masked_invalid(array)
            valid = np.ma.compressed(array)
            return float(np.mean(valid)) if len(valid) > 0 else None
    except Exception as exc:
        print(f"    Error: {exc}")
        return None


def compute_area_ha(gdf_wetland: gpd.GeoDataFrame) -> float:
    centroid = gdf_wetland.geometry.union_all().centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = "south" if centroid.y < 0 else "north"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    projected = gdf_wetland.to_crs(epsg=epsg)
    return round(float(projected.geometry.area.sum()) / 10_000, 1)


def get_tiff_path(source_cfg: dict, wetland_id: str, year: int, season: str, index_key: str) -> Path | None:
    for root in get_source_input_roots(source_cfg):
        for ext in (".tif", ".TIF"):
            candidate = root / wetland_id / f"{year}_{season}_{index_key}{ext}"
            if candidate.exists():
                return candidate
    return None


def mean_or_none(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def compute_trend(points: list[dict]) -> dict | None:
    annual_map = {}
    for point in points:
        annual_map.setdefault(point["year"], []).append(point["value"])

    annual_summary = [
        {"year": year, "mean": round(float(np.mean(values)), 4)}
        for year, values in sorted(annual_map.items())
    ]
    if len(annual_summary) < 2:
        return None

    years = np.array([row["year"] for row in annual_summary], dtype=float)
    means = np.array([row["mean"] for row in annual_summary], dtype=float)
    slope, intercept = np.polyfit(years, means, 1)
    return {
        "annual_summary": annual_summary,
        "slope_per_year": round(float(slope), 4),
        "direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
        "line": [
            {"year": row["year"], "value": round(float((slope * row["year"]) + intercept), 4)}
            for row in annual_summary
        ],
    }


def compute_metrics(points: list[dict]) -> dict:
    if not points:
        return {
            "latest": None,
            "historical_mean": None,
            "seasonal_means": {},
            "annual_summary": [],
            "trend": None,
        }

    sorted_points = sorted(points, key=lambda item: item["date"])
    latest = sorted_points[-1]
    all_values = [point["value"] for point in sorted_points]
    historical_mean = mean_or_none(all_values)

    seasonal_groups = {}
    seasonal_labels = {}
    for point in sorted_points:
        seasonal_groups.setdefault(point["season_key"], []).append(point["value"])
        seasonal_labels.setdefault(point["season_key"], point["season_label"])
    seasonal_means = {
        key: {"label": seasonal_labels[key], "mean": mean_or_none(values)}
        for key, values in seasonal_groups.items()
    }

    seasonal_mean = seasonal_means.get(latest["season_key"], {}).get("mean")
    absolute_anomaly = round(latest["value"] - historical_mean, 4) if historical_mean is not None else None
    percent_anomaly = (
        round((absolute_anomaly / abs(historical_mean)) * 100, 2)
        if historical_mean not in (None, 0)
        else None
    )
    seasonal_abs = round(latest["value"] - seasonal_mean, 4) if seasonal_mean is not None else None
    seasonal_pct = (
        round((seasonal_abs / abs(seasonal_mean)) * 100, 2)
        if seasonal_mean not in (None, 0)
        else None
    )
    trend = compute_trend(sorted_points)

    latest_payload = {
        "date": latest["date"],
        "label": latest["label"],
        "year": latest["year"],
        "season_key": latest["season_key"],
        "season_label": latest["season_label"],
        "value": latest["value"],
        "historical_mean": historical_mean,
        "historical_anomaly_abs": absolute_anomaly,
        "historical_anomaly_pct": percent_anomaly,
        "seasonal_mean": seasonal_mean,
        "seasonal_anomaly_abs": seasonal_abs,
        "seasonal_anomaly_pct": seasonal_pct,
    }

    return {
        "latest": latest_payload,
        "historical_mean": historical_mean,
        "seasonal_means": seasonal_means,
        "annual_summary": trend["annual_summary"] if trend else [],
        "trend": trend,
    }


def export_source(
    config: dict,
    source_key: str,
    source_cfg: dict,
    gdf_aoi: gpd.GeoDataFrame,
    years: list[int],
) -> dict:
    id_col = config["shapefile_id_col"]
    indices_cfg = config["indices"]
    visualization_cfg = config.get("raster_visualization", {})
    source_output_dir = Path(source_cfg["static_root"])
    rasters_dir = source_output_dir / "rasters"
    csv_dir = source_output_dir / "csv"
    source_output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    source_timeseries = {
        "source": {
            "id": source_key,
            "label": source_cfg["label"],
            "description": source_cfg.get("description", ""),
            "has_data": False,
        },
        "wetlands": {},
    }
    rasters_index = {}
    wetlands_info = {}
    periods_set = set()
    total_points = 0

    print(f"\nFuente: {source_cfg['label']} ({source_key})")

    for wetland_id, wetland_cfg in config["wetlands"].items():
        wetland_name = wetland_cfg.get("name", wetland_id)
        gdf_w = gdf_aoi[gdf_aoi[id_col].astype(str).str.strip().str.lower() == wetland_id.lower()]
        if gdf_w.empty:
            print(f"  [WARN] Sin polígonos para {wetland_id}")
            wetlands_info[wetland_id] = {
                "name": wetland_name,
                "area_ha": None,
                "center": None,
                "n_periods": 0,
                "last_period": None,
                "available_years": [],
                "status": "missing_aoi",
            }
            continue

        area_ha = compute_area_ha(gdf_w)
        geom_union = mapping(gdf_w.geometry.union_all())
        center = gdf_w.geometry.union_all().centroid
        print(f"  - {wetland_name} ({wetland_id}) | {area_ha} ha")

        wetland_entry = {"name": wetland_name, "indices": {}}
        wetland_periods = set()

        for index_key, index_cfg in indices_cfg.items():
            points = []
            for year in years:
                for season_key, season_label in config["seasons"].items():
                    tiff_path = get_tiff_path(source_cfg, wetland_id, year, season_key, index_key)
                    if tiff_path is None:
                        continue

                    mean_value = extract_zonal_mean(tiff_path, geom_union)
                    if mean_value is None:
                        continue

                    period_key = f"{year}_{season_key}"
                    label = f"{season_label} {year}"
                    point = {
                        "date": f"{year}{SEASON_MID_DATES[season_key]}",
                        "label": label,
                        "year": year,
                        "season_key": season_key,
                        "season_label": season_label,
                        "period_key": period_key,
                        "value": round(mean_value, 4),
                    }
                    points.append(point)
                    wetland_periods.add((period_key, label))
                    periods_set.add((period_key, label))

                    preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
                    preview_name = f"{wetland_id}_{year}_{season_key}_{index_key}.{preview_ext}"
                    preview_meta = build_preview_raster(
                        tiff_path,
                        rasters_dir / preview_name,
                        index_cfg["colormap"],
                        index_cfg["vmin"],
                        index_cfg["vmax"],
                        visualization_cfg,
                    )
                    rasters_index[f"{wetland_id}_{period_key}_{index_key}"] = {
                        "source": source_key,
                        "wetland_id": wetland_id,
                        "index": index_key,
                        "year": year,
                        "season": season_label,
                        "period_key": period_key,
                        "visual": preview_meta,
                        "analytic_path": Path(tiff_path).as_posix(),
                    }

            cleaned_points = [dict(point) for point in sorted(points, key=lambda item: item["date"])]
            metrics = compute_metrics(cleaned_points)
            wetland_entry["indices"][index_key] = {
                "points": cleaned_points,
                "metrics": metrics,
            }
            total_points += len(cleaned_points)

        source_timeseries["wetlands"][wetland_id] = wetland_entry
        latest_period = None
        if wetland_periods:
            latest_period = sorted(wetland_periods, key=lambda item: item[0])[-1][1]
        wetlands_info[wetland_id] = {
            "name": wetland_name,
            "area_ha": area_ha,
            "center": [round(center.y, 4), round(center.x, 4)],
            "n_periods": len(wetland_periods),
            "last_period": latest_period,
            "available_years": sorted({int(item[0].split("_")[0]) for item in wetland_periods}),
            "status": "ready" if wetland_periods else "empty",
        }

    source_timeseries["source"]["has_data"] = total_points > 0

    for wetland_id, wetland_entry in source_timeseries["wetlands"].items():
        rows = []
        for index_key, index_entry in wetland_entry["indices"].items():
            for point in index_entry["points"]:
                rows.append(
                    {
                        "source": source_key,
                        "wetland_id": wetland_id,
                        "wetland_name": wetland_entry["name"],
                        "index": index_key.upper(),
                        "date": point["date"],
                        "year": point["year"],
                        "season": point["season_label"],
                        "period_key": point["period_key"],
                        "mean": point["value"],
                    }
                )
        if rows:
            csv_path = csv_dir / f"{wetland_id}_timeseries.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source",
                        "wetland_id",
                        "wetland_name",
                        "index",
                        "date",
                        "year",
                        "season",
                        "period_key",
                        "mean",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    sorted_periods = [{"key": key, "label": label} for key, label in sorted(periods_set, key=lambda item: item[0])]
    summary = {
        "n_wetlands": len(config["wetlands"]),
        "n_periods": len(sorted_periods),
        "data_points": total_points,
        "total_area_ha": round(
            sum(info["area_ha"] for info in wetlands_info.values() if info["area_ha"] is not None),
            1,
        ),
        "latest_period": sorted_periods[-1]["label"] if sorted_periods else None,
        "available_years": sorted({int(item["key"].split("_")[0]) for item in sorted_periods}),
    }
    source_metadata = {
        "generated_at": iso_now(),
        "aoi_id_column": id_col,
        "source": {
            "id": source_key,
            "label": source_cfg["label"],
            "description": source_cfg.get("description", ""),
            "color": source_cfg.get("color"),
            "has_data": total_points > 0,
        },
        "indices": {
            key: {
                "label": value["label"],
                "description": value["description"],
                "vmin": value["vmin"],
                "vmax": value["vmax"],
                "colormap": value["colormap"],
            }
            for key, value in indices_cfg.items()
        },
        "wetlands": wetlands_info,
        "rasters": rasters_index,
        "periods": sorted_periods,
        "summary": summary,
    }

    json_dump(source_output_dir / "timeseries.json", source_timeseries)
    json_dump(source_output_dir / "metadata.json", source_metadata)

    print(f"  Timeseries -> {(source_output_dir / 'timeseries.json').as_posix()}")
    print(f"  Metadata   -> {(source_output_dir / 'metadata.json').as_posix()}")

    rel_source_dir = source_output_dir.relative_to(OUTPUT_DIR).as_posix()
    return {
        "id": source_key,
        "label": source_cfg["label"],
        "description": source_cfg.get("description", ""),
        "color": source_cfg.get("color"),
        "has_data": total_points > 0,
        "timeseries_path": f"{rel_source_dir}/timeseries.json",
        "metadata_path": f"{rel_source_dir}/metadata.json",
        "csv_dir": f"{rel_source_dir}/csv",
        "status": "ready" if total_points > 0 else "empty",
        "summary": summary,
    }


def export_static_data() -> None:
    print("=" * 60)
    print("Exportación de datos – Humedales Altoandinos")
    print("=" * 60)

    config = load_config()
    years = resolve_years(config)
    master_aoi_path = ensure_master_aoi(config)
    id_col = config["shapefile_id_col"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf_aoi = load_aoi(master_aoi_path, id_col)
    selected_ids = {key.lower() for key in config["wetlands"].keys()}
    gdf_aoi = gdf_aoi[gdf_aoi[id_col].astype(str).str.strip().str.lower().isin(selected_ids)].copy()
    print(f"AOIs cargados: {gdf_aoi[id_col].unique().tolist()}")

    aoi_output_path = OUTPUT_DIR / "wetlands_aoi.geojson"
    gdf_aoi.to_file(aoi_output_path, driver="GeoJSON")
    print(f"GeoJSON maestro -> {aoi_output_path.as_posix()}")

    manifest = {
        "generated_at": iso_now(),
        "aoi_path": aoi_output_path.relative_to(OUTPUT_DIR).as_posix(),
        "year_range": {"start": years[0], "end": years[-1]},
        "sources": {},
    }

    for source_key, source_cfg in config.get("sources", {}).items():
        if not source_cfg.get("enabled", True):
            continue
        manifest["sources"][source_key] = export_source(config, source_key, source_cfg, gdf_aoi, years)

    json_dump(OUTPUT_DIR / "sources_manifest.json", manifest)
    print(f"\nManifest -> {(OUTPUT_DIR / 'sources_manifest.json').as_posix()}")
    print("\n--- Exportación finalizada ---")


if __name__ == "__main__":
    export_static_data()
