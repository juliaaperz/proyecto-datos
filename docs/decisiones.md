# Registro de decisiones

Anotad aquí cada decisión de modelado que tenga que justificarse en el README y en la defensa
(qué se decidió, por qué y quién). Es la materia prima de la sección de Metodología.

## D1. Regla de etiquetado: «primer código» (semana 1)

El enunciado admite dos lecturas:

| Regla | Qué hace | Proyectos conservados |
|---|---|---|
| `first` (**elegida**) | Se usa solo el primer código del proyecto; si es una raíz (no llega al nivel 2), el proyecto se descarta | 31.951 (**90,313 %**) |
| `first_valid` | Se recorre la lista hasta el primer código que llegue al nivel 2 | 31.983 (90,401 %) |

Elegimos `first` porque reproduce exactamente el punto de control del enunciado (90,31 %), así
que el problema es el mismo que el de los demás equipos. La diferencia es de 32 proyectos (0,1 %)
y no afecta a ninguna conclusión. Pendiente: confirmarlo con el profesor. Para cambiar de regla:
`python scripts/build_dataset.py --rule first_valid --force-split`.

## D2. Taxonomía

* El nivel de un término se obtiene de `full_code` (`/27/79/483` → raíz 27, nivel 2 = 79).
* Se ignoran las 3 filas de `SciVocCodes.xlsx` sin `code`. Ningún proyecto las usa y una de ellas
  es *other natural sciences*, que por tanto no puede ser clase.
* Todos los códigos de los proyectos están en `SciVocCodes.xlsx`. Hay 251 apariciones de códigos
  raíz, que no alcanzan el nivel 2.
* Resultado: 39 clases de nivel 2 y 6 de nivel 1. *Humanities*, *agricultural sciences*, etc. no
  aparecen como raíces vacías.

## D3. Partición train / val / test (semana 1)

* 70 / 15 / 15, estratificada por la etiqueta de nivel 2, semilla 42.
* **Agrupada por resumen.** 40 proyectos comparten resumen con otro (p. ej., los de la *Enterprise
  Europe Network*). Se asignan siempre a la misma partición para evitar fuga entre train y test.
* La partición está en `splits/split.csv` y se sube a git. **Nadie la regenera sin avisar.**
* La misma partición se usa para el nivel 1: está estratificada por el nivel 2, así que también lo
  está, de forma aproximada, por el nivel 1.
* La clase más pequeña (*other medical sciences*, 9 proyectos) solo tiene 1 o 2 ejemplos en val y
  en test. Su F1 por clase será muy inestable; hay que decirlo al interpretar el macro-F1.
* Hiperparámetros: se ajustan con **val** (o con validación cruzada sobre train). **Test** solo se
  usa una vez por configuración final.

## D4. Columnas con riesgo de fuga (a discutir)

`projects.xlsx` trae columnas que no aparecen en el enunciado: `topic`, `topic_title`,
`masterCall`, `subCall`, `fundingScheme` y `rcn`. `topic` y `topic_title` describen la
convocatoria y están muy ligadas a la temática. **No deben usarse como entrada de los modelos de
texto del proyecto básico.** Pueden ser interesantes para la extensión de metadatos, pero hay que
avisar de que se acercan mucho a la etiqueta.

## Otras observaciones de los datos

* 3.326 proyectos no tienen ningún código EuroSciVoc.
* Algunos resúmenes son anómalos: el más corto tiene 1 palabra. Revisarlo en el EDA y decidir.

## D5. Preprocesado: lematización con spaCy (semana 2)

Se lematiza en lugar de hacer stemming porque devuelve palabras reales, lo que hace los términos y
pesos interpretables. Se quitan las stopwords de spaCy y los tokens no alfabéticos o de menos de 3
letras. Las «stopwords del dominio» se dejan a `max_df`. Se procesa solo `summary`, no `title`, que
podría añadirse como mejora. Resultado: mediana de 163 lemas por documento y 1 documento vacío.

## D6. Clasificador sklearn: SVM lineal en lugar de regresión logística (semana 2)

La regresión logística con `lbfgs` tarda unos 90 s por ajuste. Con `saga` y
`class_weight="balanced"` no converge en minutos. Con pesos uniformes da el mismo macro-F1 en val
(0,36) que `LinearSVC`, que tarda unos 5 s. Se elige la SVM. Consecuencia: no hay probabilidades,
solo puntuaciones (`decision_function`); para la evaluación basta con el argmax.

## D7. TF-IDF + SVM: hiperparámetros elegidos (semana 2)

Unigramas, `min_df=5`, `max_df=0,5`, `C=0,1`, `class_weight="balanced"`. Macro-F1 en val 0,403.
Se eligió con una búsqueda en dos fases sobre validación (23 configuraciones distintas; ver README §3).
Criterio de selección: macro-F1 en val. Los pesos balanceados aportan +4,7 puntos; el resto de
diferencias son de 1–2 puntos, probablemente dentro del ruido de validación.

## D8. Extensión: metadatos frente a texto

Se usan coste total, contribución UE, duración en meses (días / 30,4375), país coordinador
y número de países participantes. En countryContr se cuentan países distintos, también
con contribución cero. No se usan topic, topic_title ni fundingScheme.

Se aplica log(1 + importe) a coste y contribución para reducir el efecto de extremos;
los importes negativos se tratan como ausentes. Se imputan valores numéricos con la mediana de train y se añaden indicadores de ausencia.
Los países con menos de 100 proyectos en train se agrupan en Otros; los países nuevos
también van a Otros. El one-hot, las medianas, el escalado y TF-IDF se ajustan solo con train.

Se comparan RandomForest (8 configuraciones), TF-IDF + SVM con su configuración previa
y TF-IDF + metadatos con LinearSVC (8 configuraciones). La selección utiliza macro-F1
de validación con la partición común y semilla 42. Se guardan predicciones de test sin
calcular métricas de test. Las diferencias por clase son descriptivas y deben leerse
junto con el soporte; no se interpretan como mejoras significativas sin incertidumbre.

En la combinación se divide el bloque de metadatos por la norma RMS de sus filas
de entrenamiento, para igualar su norma cuadrática media a 1, como el bloque TF-IDF.
Es un único factor calculado con train: conserva las relaciones entre variables y
evita que el tamaño del bloque determine por sí solo el peso frente al texto.

## D9. Word2Vec + SVM — Lucía Fournier

Se entrenó Word2Vec exclusivamente con los documentos de train,
utilizando los textos lematizados del preprocesamiento común,
convertidos en listas de palabras mediante split().
Se utilizó skip-gram (sg=1), vector_size=200, window=5,
min_count=5, 10 épocas, seed=42 y workers=1.
También se fijó PYTHONHASHSEED=42 antes de ejecutar el script.

Se eligió entrenar los embeddings con el corpus para aprender
relaciones entre términos propios de los proyectos de investigación.
El vocabulario obtenido contiene 22.786 palabras. No se comparó
esta primera versión con vectores preentrenados.

Cada documento se representó mediante la media simple de los
vectores de sus palabras conocidas. Se ignoraron las palabras OOV;
si no había ninguna palabra conocida, se asignó un vector de ceros.
Este promedio pierde el orden de las palabras y no representa
explícitamente su contexto dentro del documento.

El porcentaje OOV se calculó sobre las apariciones de tokens:
2,5689 % en train, 3,6424 % en validación y 3,5756 % en test.
Hubo un documento sin palabras conocidas en train y ninguno
en validación ni test. Las estadísticas de test se registraron
después de seleccionar el modelo y no se utilizaron para ajustarlo.

Se aplicó StandardScaler ajustado exclusivamente con train.
El clasificador fue LinearSVC con class_weight="balanced".
Se probaron C = 0.03, 0.1, 0.3, 1, 3 y 10, seleccionando
por macro-F1 en validación. El mejor valor entre los probados
fue C=0.03, con macro-F1 de validación de 0,2934.

Se guardaron las predicciones de validación y test con columnas
projectID, y_true e y_pred. No se calcularon métricas de test.
