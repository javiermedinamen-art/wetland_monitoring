from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import yaml
from shapely.ops import transform

CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json_if_exists(path: str | Path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_file_fingerprint(path: str | Path) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": file_path.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def resolve_years(config: dict) -> list[int]:
    years = config.get("years")
    if years:
        return [int(year) for year in years]

    start = int(config.get("year_start", datetime.now().year))
    end = config.get("year_end", datetime.now().year)
    if isinstance(end, str) and end.lower() == "auto":
        end = datetime.now().year
    end = int(end)
    if end < start:
        raise ValueError(f"Rango de años inválido: {start} > {end}")
    return list(range(start, end + 1))


def _to_2d(geometry):
    if geometry is None:
        return None
    return transform(lambda x, y, z=None: (x, y), geometry)


def ensure_master_aoi(config: dict) -> Path:
    output_path = Path(config["shapefile_path"])
    wetlands_cfg = config.get("wetlands", {})
    records = []

    for wetland_id, wetland_cfg in wetlands_cfg.items():
        aoi_source = wetland_cfg.get("aoi_source")
        if not aoi_source:
            continue

        gdf = gpd.read_file(aoi_source).to_crs("EPSG:4326")
        if gdf.empty:
            continue

        name_field = wetland_cfg.get("name_field")
        raw_name = None
        if name_field and name_field in gdf.columns:
            series = gdf[name_field].dropna().astype(str).str.strip()
            raw_name = series.iloc[0] if not series.empty else None

        records.append(
            {
                "wetland_id": wetland_id,
                "nombre": wetland_cfg.get("name") or raw_name or wetland_id,
                "fuente": str(aoi_source).replace("\\", "/"),
                "geometry": _to_2d(gdf.geometry.union_all()),
            }
        )

    if records:
        master_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        master_gdf.to_file(output_path, driver="GeoJSON")

    return output_path


def get_source_config(config: dict, source_key: str) -> dict:
    sources = config.get("sources", {})
    if source_key not in sources:
        raise KeyError(f"Fuente '{source_key}' no configurada")
    return sources[source_key]


def get_source_input_roots(source_cfg: dict) -> list[Path]:
    roots = [Path(source_cfg["input_root"])]
    roots.extend(Path(item) for item in source_cfg.get("legacy_input_roots", []))
    seen = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


def relative_posix(path: str | Path, start: str | Path = ".") -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()
