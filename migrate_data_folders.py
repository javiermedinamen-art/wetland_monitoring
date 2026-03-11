"""
Migra archivos de datos a la estructura por índice (rgb, ndvi, ndwi).
Ejecuta una sola vez para reorganizar data/sentinel2 y data/drone.
Estructura nueva: {wetland_id}/{index_key}/{year}_{season}.tif
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pipeline_utils import load_config


def migrate_source(root: Path, indices: list[str]) -> int:
    moved = 0
    for wetland_dir in root.iterdir():
        if not wetland_dir.is_dir() or wetland_dir.name.startswith("_"):
            continue
        for f in list(wetland_dir.iterdir()):
            if not f.is_file():
                continue
            stem = f.stem
            ext = f.suffix
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            season_part = "_".join(parts[1:-1]) if len(parts) > 3 else parts[1]
            file_index = parts[-1].lower()
            if file_index not in (i.lower() for i in indices):
                continue
            index_dir = wetland_dir / file_index
            index_dir.mkdir(parents=True, exist_ok=True)
            dest = index_dir / f"{year}_{season_part}{ext}"
            if dest.resolve() != f.resolve() and (not dest.exists() or dest.stat().st_size == 0):
                shutil.move(str(f), str(dest))
                print(f"  {f.relative_to(root)} -> {dest.relative_to(root)}")
                moved += 1
    return moved


def main() -> None:
    config = load_config()
    total = 0

    for source_key, source_cfg in config.get("sources", {}).items():
        if not source_cfg.get("enabled", True):
            continue
        root = Path(source_cfg["input_root"])
        if not root.exists():
            continue
        indices = source_cfg.get("indices") or list(config.get("indices", {}).keys())
        indices = [i for i in indices if i in config.get("indices", {})]
        print(f"\n=== {source_cfg['label']} ({root}) ===")
        n = migrate_source(root, indices)
        total += n

    print(f"\nMigrados {total} archivos.")


if __name__ == "__main__":
    main()
