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
_Persona A: descripción, valores ausentes, distribución de clases, longitud de resúmenes,
proyectos por año y país, términos frecuentes, nube de palabras y ejemplos por clase._

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
