"""Genera notebooks/01_eda.ipynb. Ejecutar desde la raíz del repo: python scripts/make_eda_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# 01 · Análisis exploratorio (Sección 4.1)

Responsable: **Lucía Mora**.

Antes de ejecutar este notebook hay que haber lanzado `python scripts/build_dataset.py`.

Las figuras se guardan en `figures/eda_*.png` y las tablas en `results/eda_*.csv` para poder
citarlas en el README.

**Criterio:** las secciones 1–3 describen el conjunto completo. El análisis de texto (sección 4)
se hace **solo con la partición de entrenamiento**, para no tomar ninguna decisión mirando test.""")

code("""import sys, json, re
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cordis import config
from cordis.data import load_projects, load_dataset, load_taxonomy, code_names

FIG = config.FIGURES_DIR; FIG.mkdir(exist_ok=True)
RES = config.RESULTS_DIR; RES.mkdir(exist_ok=True)

# Estilo común: ejes y rejilla discretos, marcas finas
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b0b0a8", "axes.labelcolor": "#3a3a36", "xtick.color": "#5c5c56",
    "ytick.color": "#5c5c56", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
BLUE = "#2a78d6"
# Un color fijo por raíz (nivel 1); se usa en todas las figuras para que cada área se reconozca
ROOT_COLORS = {23: "#2a78d6", 25: "#eb6834", 29: "#1baf7a", 21: "#eda100", 31: "#e87ba4", 27: "#008300"}

def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); plt.show()

raw = load_projects()
df = load_dataset()
names = code_names(load_taxonomy())
print(f"Proyectos originales: {len(raw):,} · con etiqueta válida: {len(df):,}")""")

md("## 1. Descripción general")
code("""desc = pd.DataFrame({
    "tipo": raw.drop(columns="codes").dtypes.astype(str),
    "ausentes": raw.drop(columns="codes").isna().sum(),
})
desc["% ausentes"] = (100 * desc["ausentes"] / len(raw)).round(1)
desc["ejemplo"] = [str(raw[c].dropna().iloc[0])[:50] if raw[c].notna().any() else "" for c in desc.index]
desc.to_csv(RES / "eda_variables.csv")
print(f"{raw.shape[0]:,} filas × {raw.shape[1]-1} columnas")
desc""")

md("""**Observaciones.**
- `euroSciVocCode` falta en 3.326 proyectos (9,4 %), que no pueden etiquetarse.
- `patentID` está vacío en el 97,5 % y `publicationID` en el 51 %; solo interesan para la extensión de producción científica.
- `masterCall` está vacío en el 58 %.
- Las columnas `topic` y `topic_title` describen la convocatoria y están muy ligadas a la temática. **No se usan como entrada** (ver `docs/decisiones.md`, D4).""")

md("## 2. Variable objetivo")
code("""cnt = df.groupby(["label_l2", "label_l1"]).size().rename("n").reset_index()
cnt["name"] = cnt["label_l2"].map(names)
cnt = cnt.sort_values("n")
cnt["pct"] = 100 * cnt["n"] / cnt["n"].sum()

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(cnt["name"], cnt["n"], color=cnt["label_l1"].map(ROOT_COLORS), height=0.72)
ax.set_xlabel("Proyectos"); ax.grid(axis="y", visible=False)
ax.set_title("Distribución de las 39 clases de nivel 2 (color = raíz de nivel 1)", loc="left")
for y, (n, p) in enumerate(zip(cnt["n"], cnt["pct"])):
    if y >= len(cnt) - 3 or y < 3:   # etiqueta solo los extremos
        ax.text(n + 30, y, f"{n:,} ({p:.2f} %)" if p < 1 else f"{n:,} ({p:.1f} %)", va="center", fontsize=8, color="#3a3a36")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROOT_COLORS.values()]
ax.legend(handles, [names[r] for r in ROOT_COLORS], frameon=False, loc="lower right", fontsize=8)
save(fig, "eda_class_distribution_l2")

imb = cnt["n"].max() / cnt["n"].min()
top5 = cnt.nlargest(5, "n")["pct"].sum()
print(f"Clases: {len(cnt)} · mayor: {cnt['n'].max()} · menor: {cnt['n'].min()} · ratio de desbalance {imb:.0f}:1")
print(f"Las 5 clases mayores suman el {top5:.1f} % de los proyectos; {(cnt['n'] < 100).sum()} clases tienen < 100 proyectos")""")

code("""c1 = df["label_l1"].value_counts()
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh([names[i] for i in c1.index[::-1]], c1.values[::-1],
        color=[ROOT_COLORS[i] for i in c1.index[::-1]], height=0.7)
for y, n in enumerate(c1.values[::-1]):
    ax.text(n + 80, y, f"{n:,} ({100*n/len(df):.1f} %)", va="center", fontsize=8)
ax.set_xlabel("Proyectos"); ax.grid(axis="y", visible=False)
ax.set_title("Distribución de las 6 clases de nivel 1", loc="left")
save(fig, "eda_class_distribution_l1")""")

code("""checks = json.load(open(RES / "week1_checks.json", encoding="utf-8"))
base = pd.DataFrame({
    lvl: {"clase mayoritaria": checks[f"baseline_l{n}_test"]["majority_class"].split(" - ")[1],
          "accuracy (test)": checks[f"baseline_l{n}_test"]["accuracy"],
          "macro-F1 (test)": checks[f"baseline_l{n}_test"]["macro_f1"]}
    for lvl, n in (("nivel 2", 2), ("nivel 1", 1))}).T
base.to_csv(RES / "eda_baseline.csv"); base""")

md("""### 2.1. Ambigüedad de la etiqueta

¿Cuántas áreas distintas tiene cada proyecto? Esto marca el «techo» del que habla el enunciado.""")
code("""amb = df["n_l2"].clip(upper=5).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.bar([str(i) if i < 5 else "5+" for i in amb.index], amb.values, color=BLUE, width=0.6)
for x, n in enumerate(amb.values):
    ax.text(x, n + 150, f"{100*n/len(df):.0f} %", ha="center", fontsize=8)
ax.set_xlabel("Nº de categorías de nivel 2 distintas en el proyecto"); ax.set_ylabel("Proyectos")
ax.grid(axis="x", visible=False)
ax.set_title("Solo 1 de cada 3 proyectos pertenece a una única área", loc="left")
save(fig, "eda_label_ambiguity")
print(f"Media de áreas de nivel 2 por proyecto: {df['n_l2'].mean():.2f}")""")

md("## 3. Estadísticas básicas")
code("""df["n_words"] = df["summary"].str.split().str.len()
print(df["n_words"].describe().round(1).to_string())
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(df["n_words"].clip(upper=400), bins=60, color=BLUE, rwidth=0.9)
ax.axvline(df["n_words"].median(), color="#3a3a36", lw=1, ls="--")
ax.text(df["n_words"].median() + 5, ax.get_ylim()[1] * 0.9, f"mediana = {df['n_words'].median():.0f}", fontsize=8)
ax.set_xlabel("Palabras por resumen (valores > 400 agrupados en 400)"); ax.set_ylabel("Proyectos")
ax.grid(axis="x", visible=False)
ax.set_title("Longitud de los resúmenes", loc="left")
save(fig, "eda_summary_length")
short = df[df["n_words"] < 30]
print(f"\\nResúmenes con < 30 palabras: {len(short)}")
short[["projectID", "title", "summary", "label_l2_name"]].head(10)""")

code("""n_chars = df["summary"].str.len()
print(n_chars.describe().round(0).to_string())
print(f"\\nResúmenes con más de 1.990 caracteres: {100*(n_chars > 1990).mean():.1f} %")""")
md("""Los resúmenes de CORDIS tienen un **límite de unos 2.000 caracteres**. La mediana es de 1.960
caracteres y alrededor de una cuarta parte está pegada al límite, así que muchos resúmenes pueden
estar **recortados**. Por eso la distribución de palabras cae de golpe en torno a 300. Esto afecta
a los Transformers: casi ningún resumen supera los 512 tokens, por lo que basta con truncar y no
hace falta trocear.""")
code("""# Longitud en tokens aproximada para los Transformers (~1,3 tokens por palabra en inglés técnico)
over = (df["n_words"] * 1.3 > 512).mean()
print(f"Fracción estimada de resúmenes que superan 512 tokens: {100*over:.1f} % "
      "(dato a comprobar con el tokenizador real en el bloque de Transformers)")""")

code("""years = df["startDate"].dt.year.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(years.index.astype(int).astype(str), years.values, color=BLUE, width=0.65)
ax.set_ylabel("Proyectos"); ax.grid(axis="x", visible=False)
ax.set_title("Proyectos por año de inicio", loc="left")
save(fig, "eda_projects_per_year")
print(years.to_string())""")

code("""ctry = df["coordinatorCountry"].value_counts()
top = ctry.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=BLUE, height=0.7)
for y, n in enumerate(top.values):
    ax.text(n + 40, y, f"{n:,}", va="center", fontsize=8)
ax.set_xlabel("Proyectos coordinados"); ax.grid(axis="y", visible=False)
ax.set_title("Top 15 países coordinadores", loc="left")
save(fig, "eda_projects_per_country")
print(f"Países coordinadores distintos: {ctry.size}; los 5 primeros coordinan el "
      f"{100*ctry.head(5).sum()/len(df):.1f} % de los proyectos")""")

md("""## 4. Análisis preliminar del texto (solo train)

Para el EDA se usa una tokenización sencilla: palabras alfabéticas de 3 o más letras, en minúsculas,
sin las stopwords estándar de inglés de scikit-learn. El preprocesado definitivo (spaCy y
lematización) se hace en `src/cordis/preprocess.py`.""")
code("""from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
tr = df[df["split"] == "train"].reset_index(drop=True)
cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\\b[a-zA-Z]{3,}\\b", min_df=5)
Xc = cv.fit_transform(tr["summary"])
vocab = np.array(cv.get_feature_names_out())
freq = np.asarray(Xc.sum(0)).ravel()
docfreq = np.asarray((Xc > 0).sum(0)).ravel() / Xc.shape[0]
top = pd.DataFrame({"term": vocab, "freq": freq, "doc_freq": docfreq}).nlargest(25, "freq")
print(f"Vocabulario (min_df=5): {len(vocab):,} términos")

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top["term"][::-1], top["freq"][::-1], color=BLUE, height=0.7)
ax.set_xlabel("Apariciones en train"); ax.grid(axis="y", visible=False)
ax.set_title("25 términos más frecuentes (train)", loc="left")
save(fig, "eda_top_terms")
top.to_csv(RES / "eda_top_terms.csv", index=False)
top""")

md("""Los términos más frecuentes (*project*, *research*, *new*, *develop*…) son **stopwords del dominio**:
aparecen en casi todos los resúmenes y no distinguen clases. La columna `doc_freq` sugiere qué
`max_df` usar en TF-IDF para eliminarlos.""")

code("""print("Términos presentes en más del 30 % de los resúmenes de train:")
dd = pd.Series(docfreq, index=vocab).sort_values(ascending=False)
print(dd[dd > 0.3].round(2).to_string())""")

code("""cv2 = CountVectorizer(stop_words="english", token_pattern=r"(?u)\\b[a-zA-Z]{3,}\\b",
                      ngram_range=(2, 2), min_df=10)
X2 = cv2.fit_transform(tr["summary"])
bg = pd.Series(np.asarray(X2.sum(0)).ravel(), index=cv2.get_feature_names_out()).nlargest(20)
bg.to_frame("freq")""")

code("""from wordcloud import WordCloud
wc = WordCloud(width=1400, height=600, background_color="white", stopwords=ENGLISH_STOP_WORDS,
               colormap="Blues", max_words=150, random_state=0, collocations=False)
wc.generate_from_frequencies(dict(zip(vocab, freq)))
fig, ax = plt.subplots(figsize=(11, 4.7))
ax.imshow(wc.recolor(color_func=lambda *a, **k: BLUE if np.random.rand() < .5 else "#184f95", random_state=0),
          interpolation="bilinear")
ax.axis("off"); ax.set_title("Nube de palabras de los resúmenes (train)", loc="left")
save(fig, "eda_wordcloud")""")

md("""### 4.1. Términos característicos por clase

Para cada clase se listan los términos con mayor estadístico χ² frente al resto (calculado en train).
Si estos términos son muy distintos entre clases, una representación de bolsa de palabras ya
debería separarlas bien.""")
code("""from sklearn.feature_selection import chi2
rows = []
for c in tr["label_l2"].value_counts().index:
    y = (tr["label_l2"] == c).values
    s, _ = chi2(Xc, y)
    # solo términos más frecuentes en la clase que fuera (asociación positiva)
    pos = np.asarray(Xc[y].mean(0)).ravel() > np.asarray(Xc[~y].mean(0)).ravel()
    s = np.where(pos, np.nan_to_num(s), 0)
    rows.append({"clase": names[c], "n_train": int(y.sum()),
                 "términos característicos": ", ".join(vocab[np.argsort(-s)[:8]])})
distinct = pd.DataFrame(rows)
distinct.to_csv(RES / "eda_distinctive_terms.csv", index=False)
pd.set_option("display.max_colwidth", 120)
distinct""")

md("### 4.2. Ejemplos por clase")
code("""rng = np.random.default_rng(config.SEED)
for c in tr["label_l1"].value_counts().index:      # un ejemplo de la clase mayor de cada raíz
    l2 = tr.loc[tr["label_l1"] == c, "label_l2"].value_counts().index[0]
    ex = tr[tr["label_l2"] == l2].iloc[rng.integers(0, (tr["label_l2"] == l2).sum())]
    print(f"[{names[c]} › {names[l2]}]  {ex['title']}")
    print(f"   {ex['summary'][:300]}...")
    print(f"   otras áreas del proyecto: {[names[a] for a in ex['all_l2'][1:]] or '—'}\\n")""")

md("""## 5. Conclusiones del EDA (para el README)

_Completar tras revisar los resultados:_ desbalance, techo por ambigüedad, longitud y su
implicación para el límite de 512 tokens, stopwords del dominio y valor de `max_df`, y anomalías
detectadas en los resúmenes.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_eda.ipynb")
print("ok")
