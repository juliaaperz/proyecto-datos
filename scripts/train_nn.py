"""MLP para TF-IDF y Word2Vec.

Todo se ajusta con train.
Configuración y época elegidas por macro-F1 de val.
Test solo se utiliza para generar predicciones.

Ejemplos:
    python scripts/train_nn.py --representation tfidf
    python scripts/train_nn.py --representation w2v
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cordis import config
from cordis.data import load_dataset
from cordis.features import make_tfidf
from cordis.preprocess import load_tokens

# Reutiliza la función de tu script anterior.
from w2v_svm import document_vectors


SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5

GRID = [
    {"hidden": 256, "dropout": 0.3, "lr": 0.001},
    {"hidden": 512, "dropout": 0.3, "lr": 0.001},
    {"hidden": 256, "dropout": 0.5, "lr": 0.0003},
]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_model(input_size, n_classes, params):
    return nn.Sequential(
        nn.Linear(input_size, params["hidden"]),
        nn.ReLU(),
        nn.Dropout(params["dropout"]),
        nn.Linear(params["hidden"], n_classes),
    )


def batch_tensor(matrix, indices):
    """Con TF-IDF, convierte a denso únicamente este batch."""
    block = matrix[indices]
    if sparse.issparse(block):
        block = block.toarray()
    return torch.from_numpy(np.asarray(block, dtype=np.float32))


def predict(model, matrix):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, matrix.shape[0], BATCH_SIZE):
            x = batch_tensor(
                matrix, slice(start, start + BATCH_SIZE)
            )
            predictions.append(model(x).argmax(dim=1).numpy())

    return np.concatenate(predictions)


def evaluate(model, matrix, labels, criterion, weights, n_classes):
    """Pérdida ponderada global y macro-F1, con dropout desactivado."""
    model.eval()
    predictions = []
    loss_sum = 0.0
    weight_sum = 0.0

    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            indices = slice(start, start + BATCH_SIZE)
            x = batch_tensor(matrix, indices)
            y = torch.as_tensor(labels[indices], dtype=torch.long)

            logits = model(x)
            loss_sum += criterion(logits, y).item()
            weight_sum += weights[y].sum().item()
            predictions.append(logits.argmax(dim=1).numpy())

    score = f1_score(
        labels,
        np.concatenate(predictions),
        labels=np.arange(n_classes),
        average="macro",
        zero_division=0,
    )

    return loss_sum / weight_sum, float(score)


def load_splits(level):
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

    label = f"label_l{level}"

    if df["tokens"].isna().any():
        raise ValueError("Faltan textos preprocesados.")
    if df[label].isna().any():
        raise ValueError("Hay etiquetas vacías.")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Hay particiones desconocidas.")

    parts = {
        name: df[df["split"] == name].copy()
        for name in ("train", "val", "test")
    }

    if any(part.empty for part in parts.values()):
        raise ValueError("Alguna partición está vacía.")

    return parts, label


def prepare_features(representation, parts, level):
    """Devuelve train, val y una función para transformar test al final."""
    if representation == "tfidf":
        best_path = (
            Path(config.RESULTS_DIR) / f"tfidf_svm_l{level}_best.json"
        )
        with open(best_path, encoding="utf-8") as file:
            params = json.load(file)["best"]["vec"]

        vectorizer = make_tfidf(**params)
        x_train = vectorizer.fit_transform(parts["train"]["tokens"])
        x_val = vectorizer.transform(parts["val"]["tokens"])

        def transform_test():
            return vectorizer.transform(parts["test"]["tokens"])

        metadata = {
            "vectorizer": params,
            "input_size": x_train.shape[1],
            "scaling": "TF-IDF l2 normalization; no StandardScaler",
        }

    else:
        params = {
            "vector_size": 200,
            "window": 5,
            "min_count": 5,
            "sg": 1,
            "seed": SEED,
            "workers": 1,
            "epochs": 10,
        }

        train_tokens = parts["train"]["tokens"].map(str.split).tolist()
        val_tokens = parts["val"]["tokens"].map(str.split).tolist()

        print("Entrenando Word2Vec solo con train...", flush=True)
        w2v = Word2Vec(sentences=train_tokens, **params)

        x_train, _ = document_vectors(train_tokens, w2v.wv)
        x_val, _ = document_vectors(val_tokens, w2v.wv)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)

        def transform_test():
            test_tokens = (
                parts["test"]["tokens"].map(str.split).tolist()
            )
            matrix, _ = document_vectors(test_tokens, w2v.wv)
            return scaler.transform(matrix).astype(np.float32)

        metadata = {
            "word2vec": params,
            "input_size": x_train.shape[1],
            "document_vector": "mean; ignore OOV; zero if empty",
            "scaling": "StandardScaler fitted only on train",
        }

    return x_train, x_val, transform_test, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation", choices=["tfidf", "w2v"], required=True
    )
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    # CPU y semillas fijas.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    set_seed()

    results = Path(config.RESULTS_DIR)
    pred_dir = results / "preds"
    figures = ROOT / "figures"

    pred_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    name = f"nn_{args.representation}_l{args.level}"
    parts, label = load_splits(args.level)

    # Las clases originales son códigos como 33, 35...
    # La pérdida necesita índices consecutivos: 0, 1, 2...
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(parts["train"][label])
    y_val = encoder.transform(parts["val"][label])
    n_classes = len(encoder.classes_)

    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.tensor(
        len(y_train) / (n_classes * counts),
        dtype=torch.float32,
    )

    x_train, x_val, transform_test, feature_info = prepare_features(
        args.representation, parts, args.level
    )

    print(
        f"{name}: {x_train.shape[1]} entradas, "
        f"{n_classes} clases, CPU",
        flush=True,
    )

    # La suma se normaliza por la suma de pesos del batch.
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")

    history = []
    trials = []
    global_best = None

    for trial, params in enumerate(GRID, start=1):
        set_seed()
        rng = np.random.default_rng(SEED)

        model = make_model(x_train.shape[1], n_classes, params)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"]
        )

        best_score = -1.0
        best_state = None
        best_epoch = 0
        stale_epochs = 0

        print(f"\nConfiguración {trial}: {params}", flush=True)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = rng.permutation(len(y_train))

            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start:start + BATCH_SIZE]
                x = batch_tensor(x_train, indices)
                y = torch.as_tensor(
                    y_train[indices], dtype=torch.long
                )

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y) / weights[y].sum()

                if not torch.isfinite(loss):
                    raise RuntimeError("La pérdida no es finita.")

                loss.backward()
                optimizer.step()

            # Ambas pérdidas se miden al final de la época,
            # con el mismo modelo y con dropout desactivado.
            train_loss, _ = evaluate(
                model, x_train, y_train, criterion, weights, n_classes
            )
            val_loss, val_f1 = evaluate(
                model, x_val, y_val, criterion, weights, n_classes
            )

            history.append({
                "trial": trial,
                "epoch": epoch,
                **params,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            })

            pd.DataFrame(history).to_csv(
                results / f"{name}_history.csv", index=False
            )

            print(
                f"Época {epoch:02d}: "
                f"loss train={train_loss:.4f}, "
                f"loss val={val_loss:.4f}, "
                f"macro-F1 val={val_f1:.4f}",
                flush=True,
            )

            if val_f1 > best_score:
                best_score = val_f1
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= PATIENCE:
                print("Early stopping.", flush=True)
                break

        trials.append({
            "trial": trial,
            **params,
            "best_epoch": best_epoch,
            "val_macro_f1": best_score,
        })
        pd.DataFrame(trials).to_csv(
            results / f"{name}_search.csv", index=False
        )

        if global_best is None or best_score > global_best["score"]:
            global_best = {
                "trial": trial,
                "params": params.copy(),
                "epoch": best_epoch,
                "score": best_score,
                "state": best_state,
            }

    # Recuperar la mejor configuración y su mejor época.
    model = make_model(
        x_train.shape[1], n_classes, global_best["params"]
    )
    model.load_state_dict(global_best["state"])

    # Test se transforma después de finalizar la selección.
    x_test = transform_test()

    for split_name, matrix in (("val", x_val), ("test", x_test)):
        predicted_indices = predict(model, matrix)
        original_labels = encoder.inverse_transform(predicted_indices)

        pd.DataFrame({
            "projectID": parts[split_name]["projectID"].to_numpy(),
            "y_true": parts[split_name][label].to_numpy(),
            "y_pred": original_labels,
        }).to_csv(
            pred_dir / f"{name}_{split_name}.csv", index=False
        )

    selected = pd.DataFrame(history)
    selected = selected[selected["trial"] == global_best["trial"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        selected["epoch"], selected["train_loss"], label="Train"
    )
    axes[0].plot(
        selected["epoch"], selected["val_loss"], label="Validación"
    )
    axes[0].set_ylabel("Entropía cruzada ponderada")
    axes[0].legend()

    axes[1].plot(selected["epoch"], selected["val_macro_f1"])
    axes[1].set_ylabel("Macro-F1 de validación")

    for axis in axes:
        axis.set_xlabel("Época")
        axis.axvline(
            global_best["epoch"], color="gray", linestyle="--"
        )
        axis.grid(alpha=0.25)

    fig.suptitle(f"{name} — configuración elegida con validación")
    fig.tight_layout()
    fig.savefig(figures / f"{name}_learning_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "model": name,
        "seed": SEED,
        "device": "cpu",
        "features": feature_info,
        "classes": encoder.classes_.tolist(),
        "class_weights": weights.tolist(),
        "class_weight_rule": "n_train / (n_classes * class_count)",
        "architecture": "Linear -> ReLU -> Dropout -> Linear",
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "search_grid": GRID,
        "best": global_best["params"],
        "best_epoch": global_best["epoch"],
        "val_macro_f1": global_best["score"],
        "curve_losses": "end-of-epoch evaluation; dropout disabled",
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    with open(
        results / f"{name}_best.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"\nFinalizado: {name}")
    print(f"Mejor configuración: {global_best['params']}")
    print(f"Mejor época: {global_best['epoch']}")
    print(f"Macro-F1 val: {global_best['score']:.4f}")
    print("Predicciones guardadas en results/preds/")
    print(f"Figura: figures/{name}_learning_curves.png")
    print("No se han calculado métricas de test.")


if __name__ == "__main__":
    main()