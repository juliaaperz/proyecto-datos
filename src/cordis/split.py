"""Partición train / val / test fija y compartida por todo el equipo.

* Estratificada por la etiqueta de nivel 2 (39 clases, la menor con 9 proyectos).
* Agrupada por resumen: los proyectos con resumen idéntico (40 casos) van siempre
  a la misma partición, para que no haya fuga de información entre train y test.
* 70 / 15 / 15, semilla fija (config.SEED).

El resultado se guarda en splits/split.csv (projectID, split) y SÍ se sube a git:
así todos usamos exactamente la misma partición aunque cada uno regenere el dataset.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def make_split(df: pd.DataFrame, label_col: str = "label_l2", seed: int = config.SEED,
               val_size: float = config.VAL_SIZE, test_size: float = config.TEST_SIZE) -> pd.DataFrame:
    groups = df.groupby("summary", sort=False).agg(
        ids=("projectID", list), y=(label_col, "first")
    ).reset_index(drop=True)

    g_trval, g_test = train_test_split(
        groups, test_size=test_size, stratify=groups["y"], random_state=seed
    )
    g_train, g_val = train_test_split(
        g_trval, test_size=val_size / (1 - test_size), stratify=g_trval["y"], random_state=seed
    )
    rows = []
    for name, g in (("train", g_train), ("val", g_val), ("test", g_test)):
        rows += [(pid, name) for ids in g["ids"] for pid in ids]
    split = pd.DataFrame(rows, columns=["projectID", "split"]).sort_values("projectID")
    assert split["projectID"].is_unique and len(split) == len(df)
    return split.reset_index(drop=True)
