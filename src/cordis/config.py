"""Rutas y constantes compartidas por todo el equipo.

Los datos NO están en el repositorio. Por defecto se buscan en `data/raw/`
(dentro del repo, ignorado por git). Se puede cambiar con la variable de
entorno CORDIS_DATA_DIR, p. ej.:

    export CORDIS_DATA_DIR=/ruta/a/mis/datos          (Linux/Mac)
    set CORDIS_DATA_DIR=C:\\ruta\\a\\mis\\datos          (Windows cmd)

Esa carpeta debe contener projects.xlsx y SciVocCodes.xlsx, o bien H2020.zip
(se descomprime automáticamente la primera vez).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CORDIS_DATA_DIR", REPO_ROOT / "data" / "raw"))
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = REPO_ROOT / "splits"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

PROJECTS_FILE = "projects.xlsx"
SCIVOC_FILE = "SciVocCodes.xlsx"
ZIP_FILE = "H2020.zip"

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
SPLIT_PATH = SPLITS_DIR / "split.csv"

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
