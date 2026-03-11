"""Verifica que todos los módulos necesarios estén instalados."""
import geopandas
import rasterio
import numpy
from PIL import Image
import yaml
import requests
from pipeline_utils import load_config

print("Todos los módulos OK")
cfg = load_config()
print("Config cargado:", list(cfg.get("sources", {}).keys()))
