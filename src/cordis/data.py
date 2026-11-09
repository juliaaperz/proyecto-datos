"""Carga de datos y construcción de las etiquetas EuroSciVoc (Sección 3 del enunciado).

Taxonomía
---------
`SciVocCodes.xlsx` contiene, para cada término, su `full_code`: el camino de
códigos desde la raíz, p. ej. ``/27/79/483/1275``. Por tanto:

* nivel 1 (raíz)    = primer elemento del camino  (``27`` -> agricultural sciences)
* nivel 2           = segundo elemento del camino (``79`` -> agricultural biotechnology)

Un término "no alcanza el nivel 2" si su camino tiene longitud 1 (es una raíz).

Regla de etiquetado
-------------------
El enunciado dice: "utilizad la primera categoría EuroSciVoc disponible para la
que pueda extraerse un objetivo válido de segundo nivel" y da como punto de
control que se conserva ~90,31 % de los proyectos. Comprobado con los datos:

* ``rule="first"``        -> se mira SOLO el primer código; si es una raíz se
                              descarta el proyecto.          -> 90,313 % (coincide)
* ``rule="first_valid"``  -> primer código de la lista que llegue a nivel 2.
                              -> 90,401 % (32 proyectos más)

Usamos ``"first"`` por defecto porque reproduce el punto de control y, por tanto,
el mismo problema que el resto de equipos. Ver docs/decisiones.md.
"""
from __future__ import annotations

import ast
import zipfile
from collections import Counter

import pandas as pd

from . import config

DEFAULT_RULE = "first"


# --------------------------------------------------------------------------- #
# Lectura de ficheros
# --------------------------------------------------------------------------- #
def _ensure_raw_files(data_dir=None):
    data_dir = config.DATA_DIR if data_dir is None else data_dir
    proj, voc = data_dir / config.PROJECTS_FILE, data_dir / config.SCIVOC_FILE
    if proj.exists() and voc.exists():
        return proj, voc
    zpath = data_dir / config.ZIP_FILE
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            for name in (config.PROJECTS_FILE, config.SCIVOC_FILE):
                z.extract(name, data_dir)
        return proj, voc
    raise FileNotFoundError(
        f"No encuentro {config.PROJECTS_FILE} y {config.SCIVOC_FILE} (ni {config.ZIP_FILE}) "
        f"en {data_dir}. Copiadlos ahí o definid CORDIS_DATA_DIR."
    )


def load_taxonomy(data_dir=None) -> pd.DataFrame:
    """Devuelve la taxonomía con columnas code, title, full_path, path (lista de int), depth.

    Se eliminan las 3 filas sin `code` (ningún proyecto puede referirse a ellas).
    """
    _, voc = _ensure_raw_files(data_dir)
    tax = pd.read_excel(voc)
    tax = tax[tax["code"].notna()].copy()
    tax["code"] = tax["code"].astype(int)
    tax["path"] = tax["full_code"].str.strip("/").str.split("/").apply(lambda p: [int(x) for x in p])
    tax["depth"] = tax["path"].str.len()
    assert tax["code"].is_unique
    assert (tax["path"].str[-1] == tax["code"]).all(), "full_code debería terminar en code"
    return tax.reset_index(drop=True)


def code_names(tax: pd.DataFrame) -> dict[int, str]:
    """code -> nombre legible (sirve para nivel 1 y nivel 2)."""
    return dict(zip(tax["code"], tax["title"]))


def load_projects(data_dir=None) -> pd.DataFrame:
    """Lee projects.xlsx y convierte euroSciVocCode de texto ('[1, 2]') a lista de int."""
    proj, _ = _ensure_raw_files(data_dir)
    df = pd.read_excel(proj)
    df["codes"] = df["euroSciVocCode"].apply(
        lambda s: [int(c) for c in ast.literal_eval(s)] if isinstance(s, str) else []
    )
    return df


# --------------------------------------------------------------------------- #
# Etiquetas
# --------------------------------------------------------------------------- #
def ancestor(code: int, code2path: dict[int, list[int]], level: int) -> int | None:
    """Ancestro de nivel `level` (1 = raíz) del término `code`, o None si no lo alcanza."""
    path = code2path.get(code)
    if path is None or len(path) < level:
        return None
    return path[level - 1]


def project_label(codes: list[int], code2path, level: int = 2, rule: str = DEFAULT_RULE):
    """Etiqueta única de un proyecto según la regla de la Sección 3."""
    if rule == "first":
        return ancestor(codes[0], code2path, level) if codes else None
    if rule == "first_valid":
        for c in codes:
            a = ancestor(c, code2path, level)
            if a is not None:
                return a
        return None
    if rule == "majority":  # sólo para análisis (estadística del 12 % del enunciado)
        anc = [a for a in (ancestor(c, code2path, level) for c in codes) if a is not None]
        if not anc:
            return None
        counts = Counter(anc)
        best = max(counts.values())
        return next(a for a in anc if counts[a] == best)  # empate -> el primero
    raise ValueError(f"Regla desconocida: {rule}")


def l2_to_l1_map(tax: pd.DataFrame) -> dict[int, int]:
    """Categoría de nivel 2 -> raíz. Útil para saber si dos clases son 'hermanas'."""
    return {p[1]: p[0] for p in tax["path"] if len(p) >= 2}


def build_labels(df: pd.DataFrame, tax: pd.DataFrame, rule: str = DEFAULT_RULE) -> pd.DataFrame:
    """Añade las columnas de etiqueta y descarta los proyectos sin etiqueta válida.

    Columnas añadidas:
      label_l2, label_l1          códigos (int) de la etiqueta de nivel 2 y su raíz
      label_l2_name, label_l1_name
      all_l2, all_l1              lista ordenada de TODAS las categorías del proyecto
                                  (para analizar si un error cae en una etiqueta descartada)
      n_l2                        nº de categorías de nivel 2 distintas del proyecto
    """
    code2path = dict(zip(tax["code"], tax["path"]))
    names = code_names(tax)
    out = df.copy()
    out["label_l2"] = out["codes"].apply(lambda c: project_label(c, code2path, 2, rule))
    out = out[out["label_l2"].notna()].copy()
    out["label_l2"] = out["label_l2"].astype(int)
    # La raíz se deduce de la etiqueta de nivel 2 (garantiza coherencia l1 <-> l2)
    out["label_l1"] = out["label_l2"].map(l2_to_l1_map(tax)).astype(int)
    out["label_l2_name"] = out["label_l2"].map(names)
    out["label_l1_name"] = out["label_l1"].map(names)

    def _all(codes, level):
        seen = []
        for c in codes:
            a = ancestor(c, code2path, level)
            if a is not None and a not in seen:
                seen.append(a)
        return seen

    out["all_l2"] = out["codes"].apply(lambda c: _all(c, 2))
    out["all_l1"] = out["codes"].apply(lambda c: _all(c, 1))
    out["n_l2"] = out["all_l2"].str.len()
    return out


# --------------------------------------------------------------------------- #
# Acceso cómodo para el resto del equipo
# --------------------------------------------------------------------------- #
def load_dataset(with_split: bool = True) -> pd.DataFrame:
    """Dataset ya etiquetado (generado por scripts/build_dataset.py), con columna `split`."""
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError("Ejecuta primero: python scripts/build_dataset.py")
    df = pd.read_parquet(config.DATASET_PATH)
    if with_split:
        split = pd.read_csv(config.SPLIT_PATH)
        df = df.merge(split, on="projectID", how="inner", validate="one_to_one")
    return df


def get_splits(level: int = 2, text_col: str = "summary"):
    """Devuelve (X_train, y_train), (X_val, y_val), (X_test, y_test) como Series.

    Ejemplo:
        (Xtr, ytr), (Xva, yva), (Xte, yte) = get_splits(level=2)
    """
    df = load_dataset()
    y = f"label_l{level}"
    parts = []
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        parts.append((d[text_col].reset_index(drop=True), d[y].reset_index(drop=True)))
    return tuple(parts)
