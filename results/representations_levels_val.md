# Comparación de niveles (validación)

| representation   |   level |   accuracy |   macro_f1 |   rank_among_available |
|:-----------------|--------:|-----------:|-----------:|-----------------------:|
| TF-IDF           |       1 |     0.6580 |     0.6126 |                      1 |
| TF-IDF           |       2 |     0.4933 |     0.4025 |                      1 |
| Word2Vec         |       1 |     0.6265 |     0.5742 |                      2 |
| Word2Vec         |       2 |     0.3959 |     0.2934 |                      2 |

Se comparan solo las representaciones disponibles. Las tareas tienen distinto número de clases; el aumento de F1 entre niveles no mide una mejora en una misma tarea.

Pendiente: Embeddings contextuales: faltan los modelos y sus predicciones de validación en ambos niveles.

No se han leído predicciones de test ni calculado métricas de test.
