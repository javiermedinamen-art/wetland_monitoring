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
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.mask import mask
from rasterio.warp import transform_bounds, transform_geom
from rasterio.transform import Affine, array_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import mapping, shape

from pipeline_utils import (
    build_file_fingerprint,
    ensure_master_aoi,
    get_source_input_roots,
    load_config,
    load_json_if_exists,
    resolve_years,
    write_json,
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
    write_json(path, payload)


def build_export_signature(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def reuse_existing_preview(
    existing_rasters: dict,
    raster_key: str,
    preview_path: str | Path,
    export_signature: str,
    source_fingerprint: dict,
) -> dict | None:
    existing_entry = existing_rasters.get(raster_key)
    preview_path = Path(preview_path)
    if not existing_entry or existing_entry.get("export_signature") != export_signature:
        return None
    if existing_entry.get("source_fingerprint") != source_fingerprint:
        return None
    if not preview_path.exists():
        return None
    return existing_entry.get("visual")


def build_existing_point_index(existing_timeseries: dict) -> dict:
    points_index = {}
    for wetland_id, wetland_entry in existing_timeseries.get("wetlands", {}).items():
        for index_key, index_entry in wetland_entry.get("indices", {}).items():
            for point in index_entry.get("points", []):
                period_key = point.get("period_key")
                if period_key:
                    points_index[(wetland_id, index_key, period_key)] = point
    return points_index


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


def _clip_geom_to_crs(geom_wgs84: dict, src_crs: rasterio.crs.CRS) -> dict:
    if not geom_wgs84 or str(src_crs) == "EPSG:4326":
        return geom_wgs84
    return transform_geom("EPSG:4326", src_crs, geom_wgs84)


def _metric_epsg_from_geometry(geom_wgs84) -> int:
    centroid = geom_wgs84.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = "south" if centroid.y < 0 else "north"
    return 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone


def buffer_geom_wgs84(geom_wgs84: dict, buffer_m: float) -> dict:
    if not geom_wgs84 or buffer_m <= 0:
        return geom_wgs84
    geom_shape = shape(geom_wgs84)
    buffered = (
        gpd.GeoSeries([geom_shape], crs="EPSG:4326")
        .to_crs(epsg=_metric_epsg_from_geometry(geom_shape))
        .buffer(buffer_m)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    return mapping(buffered)


def _resolve_preview_shape(width: int, height: int, max_size: int | None) -> tuple[int, int]:
    if not max_size or max_size <= 0:
        return max(1, int(height)), max(1, int(width))
    scale = min(1.0, float(max_size) / float(max(width, height)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stretch_to_uint8(data: np.ndarray, valid_mask: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    band = data.astype(np.float32, copy=False)
    valid = band[valid_mask]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    lo, hi = np.percentile(valid, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(band, 0, 255).astype(np.uint8)

    stretched = np.clip((band - lo) / (hi - lo), 0, 1) * 255.0
    return stretched.astype(np.uint8)


def _quantize_rgba(image: Image.Image, colors: int) -> Image.Image:
    if colors <= 0 or colors >= 256:
        return image
    rgb = image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    alpha = image.getchannel("A")
    return Image.merge("RGBA", (*rgb.split(), alpha))


def build_preview_rgb(
    raster_path: str | Path,
    preview_path: str | Path,
    visualization_cfg: dict,
    geom_wgs84: dict | None = None,
) -> dict:
    clip_to_aoi = visualization_cfg.get("clip_to_aoi", False) and geom_wgs84 is not None
    rgb_quality = int(visualization_cfg.get("rgb_webp_quality", 55))
    rgb_max_size = int(visualization_cfg.get("rgb_max_size", 8192))
    rgb_quantize_colors = int(visualization_cfg.get("rgb_quantize_colors", 64))
    stretch_percentiles = tuple(visualization_cfg.get("rgb_stretch_percentiles", [2, 98]))
    if len(stretch_percentiles) != 2:
        stretch_percentiles = (2, 98)

    with rasterio.open(raster_path) as src:
        count = src.count
        alpha_band = 4 if count >= 4 else None
        native_window_w = int(src.width)
        native_window_h = int(src.height)
        output_transform = src.transform
        geom_crs = None

        if clip_to_aoi:
            geom_crs = _clip_geom_to_crs(geom_wgs84, src.crs)
            try:
                window = geometry_window(src, [geom_crs], pad_x=0, pad_y=0)
                native_window_w = int(window.width)
                native_window_h = int(window.height)
                bounds = window_bounds(window, src.transform)
                output_transform = src.window_transform(window)
            except Exception:
                clip_to_aoi = False
                window = None
        else:
            window = None

        if not clip_to_aoi:
            bounds = src.bounds
            output_transform = src.transform

        out_h, out_w = _resolve_preview_shape(native_window_w, native_window_h, rgb_max_size)
        read_kwargs = {
            "out_shape": (count, out_h, out_w),
            "resampling": Resampling.bilinear if count >= 3 else Resampling.nearest,
        }
        if window is not None:
            read_kwargs["window"] = window
        raster = src.read(**read_kwargs)
        native_h, native_w = native_window_h, native_window_w

        if out_w != native_window_w or out_h != native_window_h:
            output_transform = output_transform * Affine.scale(native_window_w / out_w, native_window_h / out_h)

        inside_mask = np.ones((out_h, out_w), dtype=bool)
        if clip_to_aoi and geom_crs is not None:
            inside_mask = geometry_mask([geom_crs], out_shape=(out_h, out_w), transform=output_transform, invert=True)

        if count >= 3:
            r = _stretch_to_uint8(raster[0], inside_mask, stretch_percentiles)
            g = _stretch_to_uint8(raster[1], inside_mask, stretch_percentiles)
            b = _stretch_to_uint8(raster[2], inside_mask, stretch_percentiles)
            alpha = np.full((out_h, out_w), 255, dtype=np.uint8)
            if alpha_band is not None:
                alpha = np.clip(raster[alpha_band - 1], 0, 255).astype(np.uint8)
            alpha = np.where(inside_mask, alpha, 0).astype(np.uint8)
            rgba = np.stack([r, g, b, alpha], axis=-1)
        else:
            data = _stretch_to_uint8(raster[0], inside_mask, stretch_percentiles)
            alpha = np.where(inside_mask, 255, 0).astype(np.uint8)
            rgba = np.stack([data, data, data, alpha], axis=-1)
        if clip_to_aoi and src.crs and str(src.crs) != "EPSG:4326":
            bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
    minx, miny, maxx, maxy = bounds if isinstance(bounds, (tuple, list)) else (bounds.left, bounds.bottom, bounds.right, bounds.top)
    bbox = [[miny, minx], [maxy, maxx]]
    image = Image.fromarray(rgba, "RGBA")
    image = _quantize_rgba(image, rgb_quantize_colors)
    upscale_factor = max(1, int(visualization_cfg.get("rgb_upscale_factor", visualization_cfg.get("upscale_factor", 1))))
    if upscale_factor > 1:
        image = image.resize(
            (image.width * upscale_factor, image.height * upscale_factor),
            resample=Image.Resampling.LANCZOS,
        )
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_format = str(visualization_cfg.get("preview_format", "WEBP")).upper()
    if preview_format == "WEBP":
        save_kwargs = {"quality": rgb_quality, "method": 6}
    else:
        save_kwargs = {"optimize": True}
    image.save(preview_path, format=preview_format, **save_kwargs)
    return {
        "path": preview_path.relative_to(OUTPUT_DIR).as_posix(),
        "bounds": bbox,
        "format": preview_format,
        "native_size": [native_w, native_h],
        "display_size": [int(image.width), int(image.height)],
        "render_mode": "smooth",
        "opacity": float(visualization_cfg.get("opacity", 0.9)),
    }


def build_preview_raster(
    tiff_path: str | Path,
    preview_path: str | Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    visualization_cfg: dict,
    geom_wgs84: dict | None = None,
) -> dict:
    clip_to_aoi = visualization_cfg.get("clip_to_aoi", False) and geom_wgs84 is not None
    with rasterio.open(tiff_path) as src:
        if clip_to_aoi:
            geom_crs = _clip_geom_to_crs(geom_wgs84, src.crs)
            try:
                clipped, out_transform = mask(src, [geom_crs], crop=True, nodata=src.nodata, filled=True)
                data = clipped[0].astype("float32")
                h, w = clipped.shape[1], clipped.shape[2]
                bounds = array_bounds(h, w, out_transform)
                if src.crs and str(src.crs) != "EPSG:4326":
                    bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
                native_size = [w, h]
            except Exception:
                clip_to_aoi = False
        if not clip_to_aoi:
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
    minx, miny, maxx, maxy = bounds if isinstance(bounds, (tuple, list)) else (bounds.left, bounds.bottom, bounds.right, bounds.top)
    bbox = [[miny, minx], [maxy, maxx]]
    return {
        "path": preview_path.relative_to(OUTPUT_DIR).as_posix(),
        "bounds": bbox,
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
    epsg = _metric_epsg_from_geometry(gdf_wetland.geometry.union_all())
    projected = gdf_wetland.to_crs(epsg=epsg)
    return round(float(projected.geometry.area.sum()) / 10_000, 1)


SEASON_ALIASES = {"otono": ["otono", "otoño"]}

def get_tiff_path(source_cfg: dict, wetland_id: str, year: int, season: str, index_key: str) -> Path | None:
    for root in get_source_input_roots(source_cfg):
        season_variants = SEASON_ALIASES.get(season, [season])
        exts = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG", ".jpg", ".JPG") if index_key == "rgb" else (".tif", ".tiff", ".TIF", ".TIFF")
        for s in season_variants:
            for ext in exts:
                candidate_new = root / wetland_id / index_key / f"{year}_{s}{ext}"
                if candidate_new.exists():
                    return candidate_new
                candidate_legacy = root / wetland_id / f"{year}_{s}_{index_key}{ext}"
                if candidate_legacy.exists():
                    return candidate_legacy
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
    source_indices = source_cfg.get("indices") or list(indices_cfg.keys())
    visualization_cfg = config.get("raster_visualization", {})
    rgb_context_buffer_m = float(visualization_cfg.get("rgb_context_buffer_m", 0))
    source_output_dir = Path(source_cfg["static_root"])
    rasters_dir = source_output_dir / "rasters"
    csv_dir = source_output_dir / "csv"
    source_output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    existing_metadata_path = source_output_dir / "metadata.json"
    existing_timeseries_path = source_output_dir / "timeseries.json"
    existing_metadata = load_json_if_exists(existing_metadata_path, {})
    existing_timeseries = load_json_if_exists(existing_timeseries_path, {})
    existing_rasters = existing_metadata.get("rasters", {})
    existing_points = build_existing_point_index(existing_timeseries)

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
    total_visual_rasters = 0

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
        preview_geom_union = buffer_geom_wgs84(geom_union, rgb_context_buffer_m)
        center = gdf_w.geometry.union_all().centroid
        print(f"  - {wetland_name} ({wetland_id}) | {area_ha} ha")

        wetland_entry = {"name": wetland_name, "indices": {}}
        wetland_periods = set()

        for index_key in source_indices:
            index_cfg = indices_cfg.get(index_key, {})
            visual_only = index_cfg.get("visual_only", False)
            points = []

            for year in years:
                for season_key, season_label in config["seasons"].items():
                    tiff_path = get_tiff_path(source_cfg, wetland_id, year, season_key, index_key)
                    if tiff_path is None:
                        continue

                    if visual_only:
                        period_key = f"{year}_{season_key}"
                        label = f"{season_label} {year}"
                        wetland_periods.add((period_key, label))
                        periods_set.add((period_key, label))
                        raster_key = f"{wetland_id}_{period_key}_{index_key}"
                        preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
                        preview_name = f"{wetland_id}_{year}_{season_key}_{index_key}.{preview_ext}"
                        preview_path = rasters_dir / preview_name
                        source_fingerprint = build_file_fingerprint(tiff_path)
                        export_signature = build_export_signature(
                            {
                                "visualization": visualization_cfg,
                                "index": index_key,
                                "buffer_m": rgb_context_buffer_m,
                                "mode": "rgb_visual",
                            }
                        )
                        preview_meta = reuse_existing_preview(
                            existing_rasters,
                            raster_key,
                            preview_path,
                            export_signature,
                            source_fingerprint,
                        )
                        if preview_meta is None:
                            preview_meta = build_preview_rgb(tiff_path, preview_path, visualization_cfg, preview_geom_union)
                        rasters_index[raster_key] = {
                            "source": source_key,
                            "wetland_id": wetland_id,
                            "index": index_key,
                            "year": year,
                            "season": season_label,
                            "period_key": period_key,
                            "visual": preview_meta,
                            "analytic_path": Path(tiff_path).as_posix(),
                            "source_fingerprint": source_fingerprint,
                            "export_signature": export_signature,
                        }
                        total_visual_rasters += 1
                        continue

                    period_key = f"{year}_{season_key}"
                    label = f"{season_label} {year}"
                    raster_key = f"{wetland_id}_{period_key}_{index_key}"
                    preview_ext = str(visualization_cfg.get("preview_format", "WEBP")).lower()
                    preview_name = f"{wetland_id}_{year}_{season_key}_{index_key}.{preview_ext}"
                    preview_path = rasters_dir / preview_name
                    source_fingerprint = build_file_fingerprint(tiff_path)
                    export_signature = build_export_signature(
                        {
                            "visualization": visualization_cfg,
                            "index": index_key,
                            "colormap": index_cfg["colormap"],
                            "vmin": index_cfg["vmin"],
                            "vmax": index_cfg["vmax"],
                            "mode": "analytic_preview",
                        }
                    )
                    preview_meta = reuse_existing_preview(
                        existing_rasters,
                        raster_key,
                        preview_path,
                        export_signature,
                        source_fingerprint,
                    )
                    cached_point = existing_points.get((wetland_id, index_key, period_key))
                    if preview_meta is not None and cached_point is not None:
                        point = dict(cached_point)
                    else:
                        mean_value = extract_zonal_mean(tiff_path, geom_union)
                        if mean_value is None:
                            continue
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
                    if preview_meta is None:
                        preview_meta = build_preview_raster(
                            tiff_path,
                            preview_path,
                            index_cfg["colormap"],
                            index_cfg["vmin"],
                            index_cfg["vmax"],
                            visualization_cfg,
                            geom_union,
                        )
                    rasters_index[raster_key] = {
                        "source": source_key,
                        "wetland_id": wetland_id,
                        "index": index_key,
                        "year": year,
                        "season": season_label,
                        "period_key": period_key,
                        "visual": preview_meta,
                        "analytic_path": Path(tiff_path).as_posix(),
                        "source_fingerprint": source_fingerprint,
                        "export_signature": export_signature,
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
        has_wetland_visuals = any(key.startswith(f"{wetland_id}_") for key in rasters_index)
        wetlands_info[wetland_id] = {
            "name": wetland_name,
            "area_ha": area_ha,
            "center": [round(center.y, 4), round(center.x, 4)],
            "n_periods": len(wetland_periods),
            "last_period": latest_period,
            "available_years": sorted({int(item[0].split("_")[0]) for item in wetland_periods}),
            "status": "ready" if (wetland_periods or has_wetland_visuals) else "empty",
        }

    source_has_data = total_points > 0 or total_visual_rasters > 0
    source_timeseries["source"]["has_data"] = source_has_data

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
        "raster_count": len(rasters_index),
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
            "has_data": source_has_data,
        },
        "indices": {
            key: (
                {"label": value["label"], "description": value.get("description", ""), "visual_only": True}
                if value.get("visual_only")
                else {"label": value["label"], "description": value.get("description", ""), "visual_only": False, "vmin": value["vmin"], "vmax": value["vmax"], "colormap": value["colormap"]}
            )
            for key, value in indices_cfg.items()
            if key in source_indices
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
        "has_data": source_has_data,
        "timeseries_path": f"{rel_source_dir}/timeseries.json",
        "metadata_path": f"{rel_source_dir}/metadata.json",
        "csv_dir": f"{rel_source_dir}/csv",
        "status": "ready" if source_has_data else "empty",
        "summary": summary,
    }


def export_static_data(selected_sources: list[str] | None = None) -> None:
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

    existing_manifest = {}
    manifest_path = OUTPUT_DIR / "sources_manifest.json"
    if selected_sources and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)

    manifest = {
        "generated_at": iso_now(),
        "aoi_path": aoi_output_path.relative_to(OUTPUT_DIR).as_posix(),
        "year_range": {"start": years[0], "end": years[-1]},
        "sources": existing_manifest.get("sources", {}).copy() if selected_sources else {},
    }

    available_sources = {
        source_key: source_cfg
        for source_key, source_cfg in config.get("sources", {}).items()
        if source_cfg.get("enabled", True)
    }
    requested_sources = selected_sources or list(available_sources.keys())
    invalid_sources = [source for source in requested_sources if source not in available_sources]
    if invalid_sources:
        raise ValueError(f"Fuentes no válidas o deshabilitadas: {invalid_sources}")

    for source_key in requested_sources:
        source_cfg = available_sources[source_key]
        manifest["sources"][source_key] = export_source(config, source_key, source_cfg, gdf_aoi, years)

    json_dump(manifest_path, manifest)
    print(f"\nManifest -> {manifest_path.as_posix()}")
    print("\n--- Exportación finalizada ---")


if __name__ == "__main__":
    import sys

    export_static_data(sys.argv[1:] or None)
