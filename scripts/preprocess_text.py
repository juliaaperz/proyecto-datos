"""Preprocesa todos los resúmenes con spaCy y guarda data/processed/tokens.parquet.

Tarda ~10 min con 2 núcleos. Trabaja por bloques que guarda en disco, así que si se
interrumpe se puede relanzar y continúa donde lo dejó.

    python scripts/preprocess_text.py                  # todo
    python scripts/preprocess_text.py --max-seconds 150   # procesa bloques durante 150 s y para
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config  # noqa: E402
from cordis.data import load_dataset  # noqa: E402
from cordis.preprocess import load_nlp, preprocess  # noqa: E402

CHUNK = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--n-process", type=int, default=2)
    args = ap.parse_args()

    df = load_dataset(with_split=False)[["projectID", "summary"]]
    chunk_dir = config.PROCESSED_DIR / "tokens_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(df) + CHUNK - 1) // CHUNK
    nlp, t0 = None, time.time()

    for i in range(n_chunks):
        out = chunk_dir / f"chunk_{i:03d}.parquet"
        if out.exists():
            continue
        if time.time() - t0 > args.max_seconds:
            print("Tiempo agotado; relanza el script para continuar.")
            break
        nlp = nlp or load_nlp()
        part = df.iloc[i * CHUNK:(i + 1) * CHUNK]
        toks = preprocess(part["summary"].tolist(), nlp=nlp, n_process=args.n_process)
        pd.DataFrame({"projectID": part["projectID"].values, "tokens": toks}).to_parquet(out, index=False)
        print(f"bloque {i + 1}/{n_chunks} ({time.time() - t0:.0f} s)", flush=True)

    done = sorted(chunk_dir.glob("chunk_*.parquet"))
    if len(done) == n_chunks:
        tok = pd.concat([pd.read_parquet(p) for p in done], ignore_index=True)
        assert len(tok) == len(df) and (tok["projectID"].values == df["projectID"].values).all()
        tok.to_parquet(config.PROCESSED_DIR / "tokens.parquet", index=False)
        n = tok["tokens"].str.split().str.len()
        print(f"OK: {len(tok)} documentos · tokens por documento: mediana {n.median():.0f}, "
              f"mín {n.min()} · documentos vacíos: {(n == 0).sum()}")
    else:
        print(f"Bloques hechos: {len(done)}/{n_chunks}")


if __name__ == "__main__":
    main()
