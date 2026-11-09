"""Tests rápidos de la lógica de etiquetado (no necesitan los datos reales).

Ejecutar con:  python -m pytest -q
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis.data import ancestor, build_labels, project_label  # noqa: E402
from cordis.split import make_split  # noqa: E402

# Mini-taxonomía: raíz 27 con hijos de nivel 2 (79, 81); raíz 25 con hijo 73
C2P = {27: [27], 79: [27, 79], 483: [27, 79, 483], 81: [27, 81],
       1275: [27, 79, 483, 1275], 25: [25], 73: [25, 73]}
TAX = pd.DataFrame({"code": list(C2P), "path": list(C2P.values()),
                    "title": [f"t{c}" for c in C2P]})


def test_ancestor():
    assert ancestor(1275, C2P, 2) == 79
    assert ancestor(1275, C2P, 1) == 27
    assert ancestor(79, C2P, 2) == 79          # un nodo de nivel 2 es su propio ancestro
    assert ancestor(27, C2P, 2) is None        # la raíz no alcanza el nivel 2
    assert ancestor(999, C2P, 2) is None       # código desconocido


def test_rules():
    codes = [27, 81, 483]
    assert project_label(codes, C2P, 2, "first") is None          # primer código es raíz
    assert project_label(codes, C2P, 2, "first_valid") == 81
    assert project_label([483, 81, 81], C2P, 2, "majority") == 81
    assert project_label([], C2P, 2, "first") is None


def test_build_labels():
    df = pd.DataFrame({"projectID": [1, 2, 3, 4],
                       "codes": [[1275, 73], [27, 81], [], [73, 25]]})
    out = build_labels(df, TAX, rule="first")
    assert list(out["projectID"]) == [1, 4]
    assert list(out["label_l2"]) == [79, 73]
    assert list(out["label_l1"]) == [27, 25]
    assert out.iloc[0]["all_l2"] == [79, 73] and out.iloc[0]["n_l2"] == 2


def test_split_groups_duplicates():
    n = 200
    df = pd.DataFrame({"projectID": range(n),
                       "summary": [f"s{i}" for i in range(n - 2)] + ["dup", "dup"],
                       "label_l2": [i % 4 for i in range(n)]})
    sp = make_split(df).merge(df, on="projectID")
    assert sp.loc[sp.summary == "dup", "split"].nunique() == 1
    assert set(sp["split"]) == {"train", "val", "test"}
