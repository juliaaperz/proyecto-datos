"""Word2Vec + SVM lineal.

Entrenamiento solo con train.
Selección de C por macro-F1 en val.
Test se utiliza únicamente para guardar predicciones.

Uso:
    python scripts/w2v_svm.py --level 2
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cordis import config
from cordis.data import load_dataset
from cordis.preprocess import load_tokens


SEED = 42
VECTOR_SIZE = 200
C_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def document_vectors(documents, keyed_vectors):
    """Media de palabras conocidas; ceros si no hay ninguna."""
    matrix = np.zeros(
        (len(documents), VECTOR_SIZE),
        dtype=np.float32,
    )

    total_tokens = 0
    oov_tokens = 0
    empty_documents = 0

    for i, tokens in enumerate(documents):
        known = [token for token in tokens if token in keyed_vectors]

        total_tokens += len(tokens)
        oov_tokens += len(tokens) - len(known)

        if known:
            matrix[i] = np.mean(keyed_vectors[known], axis=0)
        else:
            empty_documents += 1

    stats = {
        "documents": len(documents),
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "oov_percent": (
            100.0 * oov_tokens / total_tokens
            if total_tokens else None
        ),
        "documents_without_known_tokens": empty_documents,
        "documents_without_known_tokens_percent": (
            100.0 * empty_documents / len(documents)
            if documents else None
        ),
    }

    return matrix, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # Cargar los datos y comprobar que no se pierde ningún proyecto.
    dataset = load_dataset()
    tokens = load_tokens().rename("tokens")

    if dataset["projectID"].duplicated().any():
        raise ValueError("Hay projectID duplicados en el dataset.")

    if tokens.index.duplicated().any():
        raise ValueError("Hay projectID duplicados en los tokens.")

    df = dataset.merge(
        tokens,
        left_on="projectID",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    if df["tokens"].isna().any():
        raise ValueError(
            "Faltan textos preprocesados. "
            "Ejecuta scripts/preprocess_text.py."
        )

    label = f"label_l{args.level}"

    if label not in df.columns:
        raise ValueError(f"No existe la columna {label}.")

    if df[label].isna().any():
        raise ValueError("Hay proyectos sin etiqueta.")

    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Hay particiones desconocidas o vacías.")

    splits = {
        name: df[df["split"] == name].copy()
        for name in ("train", "val", "test")
    }

    if any(part.empty for part in splits.values()):
        raise ValueError("Alguna partición está vacía.")

    # load_tokens devuelve cadenas; Word2Vec necesita listas de palabras.
    documents = {
        name: part["tokens"].map(str.split).tolist()
        for name, part in splits.items()
    }

    train = splits["train"]
    val = splits["val"]

    classes = np.unique(train[label].to_numpy())

    if not val[label].isin(classes).all():
        raise ValueError("Val contiene clases ausentes en train.")

    print(
        f"train {len(train)} · val {len(val)} · "
        f"test {len(splits['test'])} · nivel {args.level}",
        flush=True,
    )

    w2v_parameters = {
        "vector_size": VECTOR_SIZE,
        "window": 5,
        "min_count": 5,
        "sg": 1,
        "seed": SEED,
        "workers": 1,
        "epochs": 10,
    }

    print("Entrenando Word2Vec solo con train...", flush=True)

    w2v = Word2Vec(
        sentences=documents["train"],
        **w2v_parameters,
    )

    # Preparar únicamente train y val durante la selección.
    x_train, train_stats = document_vectors(
        documents["train"], w2v.wv
    )
    x_val, val_stats = document_vectors(
        documents["val"], w2v.wv
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    results_dir = Path(config.RESULTS_DIR)
    pred_dir = results_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    name = f"w2v_svm_l{args.level}"
    search_rows = []
    best_model = None
    best_c = None
    best_f1 = -1.0

    for c in C_VALUES:
        model = LinearSVC(
            C=c,
            class_weight="balanced",
            random_state=SEED,
            max_iter=10000,
        )

        model.fit(x_train, train[label])
        predictions = model.predict(x_val)

        score = f1_score(
            val[label],
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        search_rows.append({
            "C": c,
            "class_weight": "balanced",
            "val_macro_f1": float(score),
        })

        pd.DataFrame(search_rows).to_csv(
            results_dir / f"w2v_svm_search_l{args.level}.csv",
            index=False,
        )

        print(
            f"C={c}: macro-F1 val = {score:.4f}",
            flush=True,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_c = c
            best_model = model

    # Modelo ya elegido: transformar test sin ajustar nada.
    x_test, test_stats = document_vectors(
        documents["test"], w2v.wv
    )
    x_test = scaler.transform(x_test)

    for split_name, matrix in (("val", x_val), ("test", x_test)):
        part = splits[split_name]

        pd.DataFrame({
            "projectID": part["projectID"].to_numpy(),
            "y_true": part[label].to_numpy(),
            "y_pred": best_model.predict(matrix),
        }).to_csv(
            pred_dir / f"{name}_{split_name}.csv",
            index=False,
        )

    summary = {
        "model": name,
        "seed": SEED,
        "level": args.level,
        "word2vec": w2v_parameters,
        "vocabulary_size": len(w2v.wv),
        "document_vector": "mean of known token vectors",
        "oov_policy": "ignore; zero vector if no known tokens",
        "scaler": "StandardScaler fitted only on train",
        "best": {
            "C": best_c,
            "class_weight": "balanced",
        },
        "val": {"macro_f1": best_f1},
        "classes": best_model.classes_.tolist(),
        "token_statistics": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
    }

    with open(
        results_dir / f"{name}_best.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(f"\nMejor C: {best_c}")
    print(f"Macro-F1 en val: {best_f1:.4f}")
    print(f"Resultados guardados en: {results_dir}")
    print("No se han calculado métricas de test.")


if __name__ == "__main__":
    main()