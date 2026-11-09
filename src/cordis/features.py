"""Representaciones vectoriales de los documentos.

Regla de oro (Apéndice A.4): todo lo que se estima a partir de los datos (vocabulario,
IDF, embeddings, scalers) se ajusta SOLO con train y después se aplica a val y test.
Por eso aquí se devuelven objetos sin ajustar, para meterlos en un Pipeline.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def make_tfidf(min_df=5, max_df=0.5, ngram_range=(1, 1), max_features=None,
               sublinear_tf=True) -> TfidfVectorizer:
    """TF-IDF sobre texto YA preprocesado (lemas separados por espacios, ver preprocess.py).

    * tf sublineal (1 + log tf): un término repetido 10 veces no pesa 10 veces más.
    * idf suavizado de scikit-learn y normalización l2 (resúmenes largos y cortos comparables).
    * min_df elimina erratas y nombres aislados; max_df, las «stopwords del dominio».
    """
    return TfidfVectorizer(
        lowercase=False, token_pattern=r"\S+",     # el texto ya viene tokenizado y en minúsculas
        min_df=min_df, max_df=max_df, ngram_range=tuple(ngram_range),
        max_features=max_features, sublinear_tf=sublinear_tf, dtype="float32",
    )
