"""Preprocesado del texto para TF-IDF y Word2Vec (Apéndice A.1 del enunciado).

Pipeline (en este orden), aplicado SOLO a la columna `summary`:
  1. Tokenización con spaCy (`en_core_web_sm`, sin parser ni NER: no hacen falta y es más rápido).
  2. Lematización (forma de diccionario: networks/networking -> network).
  3. Minúsculas.
  4. Se eliminan puntuación, números y cualquier token no alfabético (`is_alpha`),
     tokens de 1-2 caracteres y stopwords estándar de spaCy (326 palabras en inglés).

La poda por frecuencia de documento (min_df / max_df), que elimina erratas y las
«stopwords del dominio» (project, research, new…), NO se hace aquí sino en el
vectorizador, porque depende del corpus de entrenamiento.

Este paso es independiente para cada documento (no aprende nada del corpus), así que
puede aplicarse a todo el conjunto sin fuga de información entre train y test.
"""
from __future__ import annotations

SPACY_MODEL = "en_core_web_sm"
MIN_LEN = 3


def load_nlp():
    import spacy
    return spacy.load(SPACY_MODEL, disable=["parser", "ner"])


def clean_doc(doc) -> list[str]:
    return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) >= MIN_LEN]


def preprocess(texts, nlp=None, batch_size=256, n_process=1) -> list[str]:
    """Devuelve cada documento como cadena de lemas separados por espacios."""
    nlp = nlp or load_nlp()
    return [" ".join(clean_doc(d)) for d in nlp.pipe(texts, batch_size=batch_size, n_process=n_process)]


def load_tokens():
    """Serie projectID -> texto preprocesado (generado por scripts/preprocess_text.py)."""
    import pandas as pd
    from . import config
    path = config.PROCESSED_DIR / "tokens.parquet"
    if not path.exists():
        raise FileNotFoundError("Ejecuta primero: python scripts/preprocess_text.py")
    return pd.read_parquet(path).set_index("projectID")["tokens"]
