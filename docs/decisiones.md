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
