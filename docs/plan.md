# Plan de trabajo y reparto (equipo de 4)

Entrega: **11-dic-2026, 23:59** (último commit). Presentación individual: semana del 14-dic.

## Calendario

| Semana | Fechas | Objetivo |
|---|---|---|
| 1 | 9–15 nov | ✅ Repositorio, etiquetas, partición fija y baseline (hecho) |
| 2 | 16–22 nov | EDA completo + las 3 representaciones + primer modelo sklearn |
| 3 | 23–29 nov | Red PyTorch, fine-tuning en Colab, ajuste de hiperparámetros con val |
| 4 | 30 nov–6 dic | Evaluación en test, nivel 1, incertidumbre, análisis de errores, extensión |
| 5 | 7–11 dic | README final, diapositivas, revisión cruzada, congelar el repositorio |

## Reparto

| Persona | Rama | Bloque | Entregables (código + figuras para el README) |
|---|---|---|---|
| **A · Lucía Mora** | `eda` | EDA + preprocesado + TF-IDF | `notebooks/01_eda.ipynb`; `src/cordis/preprocess.py` (spaCy, lematización, guardar tokens en caché); `src/cordis/features.py::tfidf` |
| **B · Lucía Fournier** | `w2v-nn` | Word2Vec + red PyTorch | `features.py::word2vec` (entrenado con train; OOV); `src/cordis/nn.py` (MLP sobre las 3 representaciones, early stopping con val, curvas de aprendizaje) |
| **C · María Sánchez** | `transformers` | Transformers | Embeddings congelados (pooling y truncado/troceado > 512 tokens) guardados en `.npy`; fine-tuning en Colab (`notebooks/04_finetune_colab.ipynb`) |
| **D · Julia Pérez** | `eval` | Evaluación + sklearn + extensión | `src/cordis/evaluate.py` (accuracy, macro-F1, por clase, IC por bootstrap, confusiones entre hermanas, «predicción ∈ etiquetas descartadas»); regresión logística / SVM sobre las 3 representaciones; extensión |

Julia Pérez, además, coordina el repositorio: revisa y fusiona las pull requests y mantiene el README
como documento único. Las dependencias entre bloques son estas: el preprocesado de Lucía Mora lo usa
Word2Vec (Lucía Fournier), y el módulo de evaluación de Julia lo usan todas para comparar. Por eso
ambos deberían estar listos a mitad de la semana 2.

Todas: redactar en el README la sección de su bloque y preparar sus diapositivas. **Cada persona
debe poder defender todo el proyecto**, porque la presentación es individual.

## Normas del repositorio

1. Los datos nunca van a git. Cada uno los copia en `data/raw/` (o define `CORDIS_DATA_DIR`).
2. Antes de nada: `python scripts/build_dataset.py`. Usad siempre `cordis.data.get_splits()` o
   `load_dataset()`; **nunca hagáis vuestra propia partición**.
3. Vectorizadores, Word2Vec y scalers se ajustan **solo con train** (usad `sklearn.pipeline.Pipeline`).
4. Test no se toca para decidir nada. Cada resultado en test se guarda en `results/*.json` con
   la configuración usada.
5. Las figuras que vayan al README se guardan en `figures/` (png) desde código, no capturas.
6. Una rama por bloque (`eda`, `w2v-nn`, `transformers`, `eval`) y merge por pull request
   revisado por otra persona.
   Revisiones cruzadas: Lucía Mora ↔ Lucía Fournier, María Sánchez ↔ Julia Pérez.
7. Las decisiones se anotan en `docs/decisiones.md` al tomarlas.
8. Si usáis código externo (tutoriales, Hugging Face, ChatGPT/Claude…), citadlo en el README.

## Pendiente de decidir en grupo

- [ ] Extensión. Propuesta: «¿Qué aporta el texto?» con metadatos (barata y con buena discusión),
      o zero-shot con un LLM (más original y más costosa). Decidir antes del 22-nov.
- [ ] Confirmar con el profesor la regla de etiquetado (ver D1).
- [ ] Modelo Transformer base: `bert-base-uncased`, `roberta-base` o `allenai/scibert_scivocab_uncased`.
