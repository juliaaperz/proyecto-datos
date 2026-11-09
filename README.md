# Análisis temático de proyectos H2020 (CORDIS)

Proyecto final de *Procesamiento de Datos*, Máster en Ingeniería de Telecomunicación (UC3M), curso 2026/27.

**Equipo:** Julia Pérez · Lucía Mora · Lucía Fournier · María Sánchez

> Este README es el **informe** del proyecto. Las instrucciones para reproducirlo están al final.
> _(Las secciones en cursiva son plantillas que hay que completar.)_

---

## 1. Problema y conjunto de datos

Clasificación multiclase de los resúmenes de 35.378 proyectos financiados por H2020 en su
categoría temática EuroSciVoc de **segundo nivel**.

### 1.1. Construcción de la etiqueta

Cada proyecto tiene una lista de términos EuroSciVoc. Para cada término se sube por la jerarquía
(`full_code` en `SciVocCodes.xlsx`) hasta su ancestro de nivel 2. La etiqueta del proyecto es la
del **primer** término de la lista, y el proyecto se descarta si ese término es una raíz. Ver
[docs/decisiones.md](docs/decisiones.md) (D1) para la discusión de la regla.

| | Valor |
|---|---|
| Proyectos iniciales | 35.378 |
| Sin ningún código EuroSciVoc | 3.326 |
| Proyectos con etiqueta válida | **31.951 (90,31 %)** ✔ punto de control |
| Clases de nivel 2 / nivel 1 | 39 / 6 |
| Proyectos con todas sus categorías en una misma clase (nivel 2 / nivel 1) | 32,2 % / 46,9 % |
| Etiquetas que cambiarían con la regla de mayoría (nivel 2 / nivel 1) | 12,1 % / 11,5 % |

### 1.2. Partición

70 / 15 / 15 (train / val / test), estratificada por clase de nivel 2 y agrupada por resumen
(los resúmenes duplicados nunca se reparten entre particiones): **22.366 / 4.792 / 4.793** proyectos.
Fichero fijo: `splits/split.csv`.
Tamaños y clases por partición: `splits/labels.csv`.

### 1.3. Modelo de referencia (baseline)

Clasificador que siempre predice la clase mayoritaria de train (resultados en test):

| Nivel | Clase mayoritaria | Accuracy | Macro-F1 |
|---|---|---|---|
| 2 (39 clases) | biological sciences | 0,108 | 0,005 |
| 1 (6 clases) | natural sciences | 0,391 | 0,094 |

El macro-F1 casi nulo del baseline muestra por qué la accuracy por sí sola no basta con este desbalance.

## 2. Análisis exploratorio

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Las secciones 2.1–2.3 describen el
conjunto completo. El análisis de texto (2.4) se hace solo con la partición de entrenamiento.

### 2.1. Descripción general

`projects.xlsx` tiene 35.378 filas y 19 columnas: identificadores, texto (`title`, `summary`),
fechas, importes, país coordinador, códigos EuroSciVoc, publicaciones y patentes. También incluye
columnas de la convocatoria que el enunciado no menciona (`topic`, `topic_title`, `masterCall`,
`subCall`, `fundingScheme`). Valores ausentes relevantes:

| Variable | Ausentes | Implicación |
|---|---|---|
| `euroSciVocCode` | 3.326 (9,4 %) | Proyectos sin etiqueta posible: se descartan |
| `publicationID` | 18.166 (51,3 %) | Solo relevante para la extensión de producción científica |
| `patentID` | 34.499 (97,5 %) | Ídem |
| `masterCall` | 20.358 (57,5 %) | No se usa |
| `startDate`, `endDate` | 11 | Solo afecta a la figura por año |

`summary` y `title` no tienen ausentes. `topic` y `topic_title` están muy ligadas a la temática,
por lo que **no se usan como entrada** (ver [decisiones](docs/decisiones.md), D4).

### 2.2. Variable objetivo

![Distribución nivel 2](figures/eda_class_distribution_l2.png)

* **39 clases con un desbalance fuerte.** La mayor (*biological sciences*, 3.462 proyectos)
  es 385 veces más grande que la menor (*other medical sciences*, 9). Las 5 clases mayores
  suman el 40 % de los proyectos y 7 clases tienen menos de 100.
* **Consecuencia para las métricas:** el macro-F1 da a las clases pequeñas el mismo peso que a
  las grandes, y estas tienen solo 1–5 ejemplos en test. Su F1 por clase será muy ruidoso, así
  que habrá que leerlo junto con el soporte de cada clase.
* **Nivel 1:** *natural sciences* concentra el 39 % y *agricultural sciences* solo el 2 %.

![Distribución nivel 1](figures/eda_class_distribution_l1.png)

**Modelo de referencia (baseline):**

| Nivel | Clase mayoritaria | Accuracy (test) | Macro-F1 (test) |
|---|---|---|---|
| 2 (39 clases) | biological sciences | 0,108 | 0,005 |
| 1 (6 clases) | natural sciences | 0,391 | 0,094 |

El baseline de nivel 2 acierta el 11 % y su macro-F1 es casi cero. Esto ilustra por qué la
accuracy por sí sola no basta con este desbalance.

**Ambigüedad de la etiqueta.** Cada proyecto tiene de media 2,2 áreas de nivel 2 distintas, y
solo un 32 % pertenece a una única área. En el 68 % restante, la etiqueta es simplemente la
primera de varias áreas igual de válidas. Esto fija un techo de rendimiento por debajo del 100 %.

![Ambigüedad](figures/eda_label_ambiguity.png)

Además, el notebook muestra que algunas etiquetas son discutibles incluso en su primer término.
Un proyecto sobre fases topológicas en materia condensada, que es física, aparece etiquetado como
*biological sciences*.

### 2.3. Estadísticas básicas

![Longitud](figures/eda_summary_length.png)

* **Longitud:** la mediana es de 276 palabras y unos 1.960 caracteres. CORDIS limita los
  resúmenes a unos **2.000 caracteres**, y alrededor del 25 % están pegados a ese límite, así
  que probablemente están recortados. Por eso la distribución cae de golpe en torno a 300
  palabras.
* **Implicación para los Transformers:** con este límite, casi ningún resumen supera los 512
  tokens. Truncar es suficiente y no hace falta trocear (hay que comprobarlo con el tokenizador).
* **Anomalías:** 25 resúmenes tienen menos de 30 palabras. Algunos son solo el título o, en un
  caso, solo la palabra «H2020». Se mantienen, pero se documentan.

| | |
|---|---|
| ![Año](figures/eda_projects_per_year.png) | ![País](figures/eda_projects_per_country.png) |

* **Proyectos por año:** unos 4.500 al año entre 2015 y 2019, menos en 2020–2021 y una cola en
  2022, que corresponde al final del programa.
* **Países:** hay 57 países coordinadores. Los 5 primeros coordinan el 57 % de los proyectos.

### 2.4. Análisis preliminar del texto (train)

![Términos frecuentes](figures/eda_top_terms.png)

![Nube de palabras](figures/eda_wordcloud.png)

* **Stopwords del dominio.** Los términos más frecuentes (*project*, *new*, *research*,
  *based*, *development*) no dicen nada de la temática. *project* aparece en el 61 % de los
  resúmenes y *new* en el 51 %. Unos 8 términos aparecen en más del 30 % de los documentos, lo
  que justifica usar `max_df` ≈ 0,3–0,5 en TF-IDF.
* **Bigramas frecuentes.** *state art*, *project aims*, *real time*, *climate change* y
  *machine learning* aparecen mucho. Algunos son temáticos, lo que da un argumento para probar
  `ngram_range=(1,2)`.
* **Términos característicos por clase.** Se calculan con χ² frente al resto de clases
  ([tabla completa](results/eda_distinctive_terms.csv)). Las clases grandes tienen un vocabulario
  muy específico: *quantum, physics, universe* en physical sciences; *cancer, patients* en
  clinical medicine; *catalysis* en chemical sciences. Esto anticipa que la bolsa de palabras
  funcionará razonablemente bien. La tabla también revela cosas de la taxonomía:
  * *social geography* es en la práctica «transporte» (*transport, airspace, ATM, SESAR*).
  * *economics and business* está dominada por los servicios a pymes (*SMEs, EEN, KAM*).
  * Las clases pequeñas tienen términos característicos poco fiables, porque salen de muy pocos
    ejemplos. *other medical sciences* (7 en train) mezcla *forensic*, *genocide* y *sperm*.

## 3. Metodología
### 3.1. Preprocesado de texto
### 3.2. Representaciones (TF-IDF, Word2Vec, embeddings contextuales)
### 3.3. Modelos (scikit-learn, red PyTorch, fine-tuning de Transformer)
### 3.4. Protocolo de validación y búsqueda de hiperparámetros

## 4. Resultados
### 4.1. Comparación representaciones × modelos (accuracy, macro-F1, IC)
### 4.2. Nivel 1 frente a nivel 2
### 4.3. Análisis de errores (confusiones entre hermanas, predicciones en etiquetas descartadas)

## 5. Extensión

## 6. Conclusiones y limitaciones

## 7. Contribuciones y reconocimiento de fuentes

---

## Reproducir los resultados

```bash
pip install -r requirements.txt
# Copiad projects.xlsx y SciVocCodes.xlsx (o H2020.zip) en data/raw/
# o definid CORDIS_DATA_DIR con la carpeta donde estén.
python scripts/build_dataset.py      # etiquetas + comprobaciones (usa splits/split.csv)
python -m pytest -q                  # tests de la lógica de etiquetado
```

En código:

```python
import sys; sys.path.insert(0, "src")
from cordis.data import get_splits, load_dataset
(Xtr, ytr), (Xva, yva), (Xte, yte) = get_splits(level=2)   # level=1 para las 6 raíces
df = load_dataset()   # todas las columnas + label_l1/l2, all_l2, n_l2, split
```

Estructura:

```
src/cordis/      código reutilizable (config, data, split, ...)
scripts/         scripts ejecutables
notebooks/       exploración y experimentos (Colab para fine-tuning)
splits/          partición común (se versiona)
results/         métricas en JSON
figures/         figuras usadas en este README
docs/            plan de trabajo y registro de decisiones
data/            datos locales (NO se versiona)
```
