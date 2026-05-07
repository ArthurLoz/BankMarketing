"""
=============================================================
  Caso de Estudio N°1 — BankMarketing EDA
  Especialización en Python for Analytics
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io

# ──────────────────────────────────────────────
# Configuración global de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BankMarketing EDA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta corporativa
PALETTE   = ["#1a73e8", "#e84040"]
PALETTE10 = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", palette=PALETTE)

# ──────────────────────────────────────────────
# POO — Clase DataAnalyzer
# ──────────────────────────────────────────────
class DataAnalyzer:
    """Encapsula estadísticas, clasificación y visualizaciones del dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ── Clasificación de variables ──────────────
    def classify_variables(self) -> dict:
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        return {"numeric": num_cols, "categorical": cat_cols}

    # ── Estadísticas descriptivas ───────────────
    def descriptive_stats(self) -> pd.DataFrame:
        num_cols = self.classify_variables()["numeric"]
        desc = self.df[num_cols].describe().T
        desc["median"] = self.df[num_cols].median()
        desc["cv_%"]   = (desc["std"] / desc["mean"] * 100).round(2)
        desc["skew"]   = self.df[num_cols].skew().round(3)
        return desc.round(3)

    # ── Valores nulos ───────────────────────────
    def missing_values(self) -> pd.DataFrame:
        total  = self.df.isnull().sum()
        pct    = (total / len(self.df) * 100).round(2)
        return pd.DataFrame({"Nulos": total, "Porcentaje (%)": pct})

    # ── Info del dataset ────────────────────────
    def dataset_info(self) -> pd.DataFrame:
        buf = io.StringIO()
        self.df.info(buf=buf)
        rows = []
        for col in self.df.columns:
            rows.append({
                "Columna":    col,
                "Tipo":       str(self.df[col].dtype),
                "No Nulos":   self.df[col].notna().sum(),
                "Únicos":     self.df[col].nunique(),
                "Ejemplo":    str(self.df[col].dropna().iloc[0]),
            })
        return pd.DataFrame(rows)

    # ── Histograma de variable numérica ─────────
    def plot_histogram(self, col: str, bins: int = 30, hue: str = None):
        fig, ax = plt.subplots(figsize=(8, 4))
        if hue:
            for val, color in zip(self.df[hue].unique(), PALETTE):
                subset = self.df[self.df[hue] == val][col].dropna()
                ax.hist(subset, bins=bins, alpha=0.6, label=f"{hue}={val}", color=color, edgecolor="white")
            ax.legend()
        else:
            ax.hist(self.df[col].dropna(), bins=bins, color=PALETTE[0], edgecolor="white")
        ax.set_title(f"Distribución de {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col); ax.set_ylabel("Frecuencia")
        plt.tight_layout()
        return fig

    # ── Boxplot ──────────────────────────────────
    def plot_boxplot(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(9, 4))
        order = self.df.groupby(cat_col)[num_col].median().sort_values().index
        sns.boxplot(data=self.df, x=cat_col, y=num_col, order=order,
                    palette="Blues_d", ax=ax, flierprops=dict(marker=".", alpha=0.3))
        ax.set_title(f"{num_col} por {cat_col}", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig

    # ── Barras categórico vs y ───────────────────
    def plot_cat_vs_target(self, cat_col: str, target: str = "y"):
        ct = (self.df.groupby([cat_col, target])
              .size().reset_index(name="count"))
        total = ct.groupby(cat_col)["count"].transform("sum")
        ct["pct"] = (ct["count"] / total * 100).round(1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        # Conteo
        sns.barplot(data=ct, x=cat_col, y="count", hue=target,
                    palette=PALETTE, ax=axes[0])
        axes[0].set_title(f"Conteo: {cat_col} vs {target}", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=30)
        # Proporción
        yes_pct = ct[ct[target] == "yes"].set_index(cat_col)["pct"]
        order   = yes_pct.sort_values(ascending=False).index
        sns.barplot(data=ct[ct[target] == "yes"], x=cat_col, y="pct",
                    order=order, color=PALETTE[0], ax=axes[1])
        axes[1].set_title(f"% Aceptación por {cat_col}", fontweight="bold")
        axes[1].set_ylabel("% yes")
        axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        return fig

    # ── Heatmap correlación ──────────────────────
    def plot_correlation(self):
        num_cols = self.classify_variables()["numeric"]
        corr = self.df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax,
                    linewidths=0.5, annot_kws={"size": 9})
        ax.set_title("Matriz de Correlación", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    # ── Pie chart ────────────────────────────────
    def plot_pie(self, col: str):
        counts = self.df[col].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel"), startangle=140,
               wedgeprops=dict(edgecolor="white"))
        ax.set_title(f"Distribución de {col}", fontweight="bold")
        return fig

    # ── Barras simples ───────────────────────────
    def plot_barh(self, col: str, top_n: int = 15):
        counts = self.df[col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.45)))
        bars = ax.barh(counts.index[::-1], counts.values[::-1],
                       color=PALETTE[0], edgecolor="white")
        for bar, val in zip(bars, counts.values[::-1]):
            ax.text(bar.get_width() + counts.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.set_title(f"Top {top_n} — {col}", fontweight="bold")
        ax.set_xlabel("Frecuencia")
        plt.tight_layout()
        return fig

    # ── Scatter dinámico ─────────────────────────
    def plot_scatter(self, x: str, y: str, hue: str = "y"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for val, color in zip(self.df[hue].unique(), PALETTE):
            sub = self.df[self.df[hue] == val]
            ax.scatter(sub[x], sub[y], alpha=0.25, s=10,
                       label=f"{hue}={val}", color=color)
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y} (color={hue})", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    # ── KDE dual ────────────────────────────────
    def plot_kde(self, col: str, hue: str = "y"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for val, color in zip(self.df[hue].unique(), PALETTE):
            sub = self.df[self.df[hue] == val][col].dropna()
            sns.kdeplot(sub, ax=ax, label=f"{hue}={val}", color=color, fill=True, alpha=0.3)
        ax.set_title(f"KDE de {col} por {hue}", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    # ── Heatmap cat vs cat ───────────────────────
    def plot_cat_heatmap(self, col1: str, col2: str):
        ct = pd.crosstab(self.df[col1], self.df[col2], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "%"})
        ax.set_title(f"{col1} vs {col2} (% por fila)", fontweight="bold")
        plt.tight_layout()
        return fig


# ──────────────────────────────────────────────
# Sidebar — Menú de navegación
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=70)
    st.title("🏦 BankMarketing EDA")
    st.markdown("---")
    menu = st.radio(
        "📌 Navegación",
        ["🏠 Home", "📂 Carga del Dataset", "🔍 Análisis EDA", "💡 Conclusiones"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Especialización Python for Analytics\nMSc. Carlos Carrillo Villavicencio")

# ──────────────────────────────────────────────
# SESSION STATE — guardar el dataframe cargado
# ──────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None

df_loaded = st.session_state["df"]
analyzer  = DataAnalyzer(df_loaded) if df_loaded is not None else None


# ══════════════════════════════════════════════
#  MÓDULO 1 — HOME
# ══════════════════════════════════════════════
if menu == "🏠 Home":
    st.title("🏦 Análisis Exploratorio de Datos — BankMarketing")
    st.markdown("### Caso de Estudio N°1 · Especialización Python for Analytics")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 Objetivo del Proyecto")
        st.markdown("""
        Este proyecto aplica de manera integrada los conceptos de **Python para Analytics**,
        desarrollando una herramienta de **Análisis Exploratorio de Datos (EDA)** sobre
        el dataset `BankMarketing.csv`, correspondiente a una campaña de marketing
        telefónico de una institución financiera portuguesa.

        El objetivo **NO** es construir modelos predictivos, sino descubrir
        **patrones, relaciones y comportamientos** en los datos que apoyen
        la toma de decisiones comerciales.
        """)

        st.subheader("🎯 Contexto del Negocio")
        st.markdown("""
        Durante los últimos 6 meses, la **efectividad de la campaña cayó del 12% al 8%**,
        afectando los bonos de los ejecutivos comerciales.

        > **Fórmula:** Efectividad = (Ventas / Base) × 100%

        El análisis busca identificar qué variables se asocian con la aceptación
        del producto bancario ofrecido (depósito a plazo).
        """)

    with col2:
        st.subheader("👤 Autor")
        st.info("""
        **Nombre:** Jimmy Arthur Lopez Ore

        **Curso:** Especialización en Python for Analytics

        **Institución:** DMC Institute

        **Año:** 2026
        """)

        st.subheader("🛠️ Tecnologías")
        techs = ["Python 3.11", "Streamlit", "Pandas", "NumPy",
                 "Matplotlib", "Seaborn", "GitHub"]
        for t in techs:
            st.markdown(f"- `{t}`")

    st.markdown("---")
    st.subheader("📊 Sobre el Dataset")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("📁 Filas",    "41,188")
    col_b.metric("📐 Columnas", "21")
    col_c.metric("🎯 Variable objetivo", "y")
    col_d.metric("📅 Fuente", "UCI ML Repo")

    st.markdown("""
    El dataset contiene información demográfica de clientes, historial crediticio,
    detalles de los contactos realizados en la campaña y variables macroeconómicas.
    La variable objetivo **`y`** indica si el cliente suscribió un depósito a plazo
    (`yes` / `no`).
    """)


# ══════════════════════════════════════════════
#  MÓDULO 2 — CARGA DEL DATASET
# ══════════════════════════════════════════════
elif menu == "📂 Carga del Dataset":
    st.title("📂 Carga del Dataset")
    st.markdown("Sube el archivo `BankMarketing.csv` para habilitar el análisis.")
    st.markdown("---")

    uploaded = st.file_uploader("📎 Selecciona el archivo CSV", type=["csv"])

    if uploaded is not None:
        try:
            sep = st.radio("Separador del CSV", [";", ",", "|"], horizontal=True)
            df  = pd.read_csv(uploaded, sep=sep)
            st.session_state["df"] = df
            df_loaded = df
            analyzer  = DataAnalyzer(df)

            st.success(f"✅ Archivo cargado correctamente: **{uploaded.name}**")
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Filas",    f"{df.shape[0]:,}")
            col2.metric("📐 Columnas", df.shape[1])
            col3.metric("🔢 Numéricas", len(df.select_dtypes(include=np.number).columns))
            col4.metric("🔤 Categóricas", len(df.select_dtypes(include="object").columns))

            st.subheader("👀 Vista previa (primeras 5 filas)")
            st.dataframe(df.head(), use_container_width=True)

            st.subheader("📋 Tipos de datos por columna")
            dtype_df = pd.DataFrame({
                "Columna": df.columns,
                "Tipo":    df.dtypes.astype(str).values,
                "No Nulos": df.notna().sum().values,
                "Únicos":  df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
    else:
        st.warning("⚠️ Ningún archivo cargado. Por favor sube el BankMarketing.csv para continuar.")


# ══════════════════════════════════════════════
#  MÓDULO 3 — ANÁLISIS EDA
# ══════════════════════════════════════════════
elif menu == "🔍 Análisis EDA":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")

    if df_loaded is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **📂 Carga del Dataset**.")
        st.stop()

    tabs = st.tabs([
        "1️⃣ Info General",
        "2️⃣ Variables",
        "3️⃣ Estadísticas",
        "4️⃣ Valores Faltantes",
        "5️⃣ Distribuciones",
        "6️⃣ Categóricas",
        "7️⃣ Bivariado Num×Cat",
        "8️⃣ Bivariado Cat×Cat",
        "9️⃣ Análisis Dinámico",
        "🔟 Hallazgos Clave",
    ])

    # ── ÍTEM 1: Información general ─────────────
    with tabs[0]:
        st.subheader("📋 Ítem 1 — Información General del Dataset")
        st.markdown("""
        En este ítem exploramos la **estructura del dataset**: tipos de datos,
        cantidad de registros, valores únicos y ausencia de nulos.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🗂️ Detalle por columna**")
            st.dataframe(analyzer.dataset_info(), use_container_width=True)
        with col2:
            st.markdown("**📊 Métricas globales**")
            st.metric("Total de registros", f"{df_loaded.shape[0]:,}")
            st.metric("Total de columnas",  df_loaded.shape[1])
            st.metric("Variables numéricas",   len(analyzer.classify_variables()["numeric"]))
            st.metric("Variables categóricas", len(analyzer.classify_variables()["categorical"]))
            st.metric("Valores nulos totales",  df_loaded.isnull().sum().sum())

            st.markdown("**🎯 Balance de la variable objetivo `y`**")
            vc = df_loaded["y"].value_counts(normalize=True).mul(100).round(1)
            for k, v in vc.items():
                color = "🟢" if k == "yes" else "🔴"
                st.markdown(f"{color} **{k}**: {v}%")

    # ── ÍTEM 2: Clasificación de variables ───────
    with tabs[1]:
        st.subheader("🗂️ Ítem 2 — Clasificación de Variables")
        st.markdown("""
        Utilizamos una **función personalizada** (método `classify_variables` de
        la clase `DataAnalyzer`) para separar automáticamente las variables
        numéricas de las categóricas según el tipo de dato del DataFrame.
        """)

        vars_dict = analyzer.classify_variables()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🔢 Variables Numéricas ({len(vars_dict['numeric'])})")
            for v in vars_dict["numeric"]:
                st.markdown(f"- `{v}`")

        with col2:
            st.markdown(f"### 🔤 Variables Categóricas ({len(vars_dict['categorical'])})")
            for v in vars_dict["categorical"]:
                st.markdown(f"- `{v}`")

        st.markdown("---")
        st.markdown("**📊 Conteo visual por tipo**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Numéricas", "Categóricas"],
               [len(vars_dict["numeric"]), len(vars_dict["categorical"])],
               color=PALETTE, edgecolor="white", width=0.4)
        ax.set_ylabel("Cantidad")
        ax.set_title("Distribución de tipos de variable", fontweight="bold")
        for i, v in enumerate([len(vars_dict["numeric"]), len(vars_dict["categorical"])]):
            ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    # ── ÍTEM 3: Estadísticas descriptivas ────────
    with tabs[2]:
        st.subheader("📈 Ítem 3 — Estadísticas Descriptivas")
        st.markdown("""
        Se aplica `.describe()` extendido con **mediana**, **coeficiente de variación (CV)**
        y **asimetría (skew)** para comprender mejor la dispersión y forma de cada variable.
        """)

        desc = analyzer.descriptive_stats()
        st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["mean","std","cv_%"]),
                     use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Interpretación rápida")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Age** — Media: {desc.loc['age','mean']:.1f} años | "
                    f"Mediana: {desc.loc['age','50%']:.0f} años")
        with col2:
            st.info(f"**Duration** — Media: {desc.loc['duration','mean']:.0f} s | "
                    f"CV: {desc.loc['duration','cv_%']:.1f}% (alta dispersión)")
        with col3:
            st.info(f"**Campaign** — Media: {desc.loc['campaign','mean']:.1f} contactos | "
                    f"Máx: {desc.loc['campaign','max']:.0f}")

    # ── ÍTEM 4: Valores faltantes ─────────────────
    with tabs[3]:
        st.subheader("🔎 Ítem 4 — Análisis de Valores Faltantes")
        st.markdown("""
        Analizamos si el dataset presenta valores nulos que requieran tratamiento
        antes de cualquier análisis o modelado.
        """)

        missing = analyzer.missing_values()
        total_missing = missing["Nulos"].sum()

        if total_missing == 0:
            st.success("✅ El dataset **no presenta valores nulos**. No se requiere imputación.")
        else:
            st.warning(f"⚠️ Se encontraron **{total_missing}** valores nulos.")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(missing, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            missing["Porcentaje (%)"].plot(kind="bar", ax=ax, color=PALETTE[0], edgecolor="white")
            ax.set_title("Porcentaje de valores nulos por columna", fontweight="bold")
            ax.set_ylabel("%")
            ax.axhline(0, color="gray", linewidth=0.5)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("""
        **💬 Discusión:** La ausencia de valores nulos indica que el dataset fue
        preprocesado previamente. Sin embargo, valores como `unknown` en variables
        categóricas (`job`, `education`, `default`, etc.) deben considerarse como
        **nulos implícitos** y tratarse según el análisis.
        """)

        # Mostrar "unknowns" implícitos
        st.markdown("#### ⚠️ Valores 'unknown' por columna (nulos implícitos)")
        cat_cols = analyzer.classify_variables()["categorical"]
        unknown_counts = {c: (df_loaded[c] == "unknown").sum()
                          for c in cat_cols if (df_loaded[c] == "unknown").any()}
        if unknown_counts:
            unk_df = pd.DataFrame.from_dict(unknown_counts, orient="index",
                                            columns=["Cantidad 'unknown'"])
            unk_df["% del total"] = (unk_df["Cantidad 'unknown'"] / len(df_loaded) * 100).round(2)
            st.dataframe(unk_df, use_container_width=True)
        else:
            st.info("No se encontraron valores 'unknown'.")

    # ── ÍTEM 5: Distribución de variables numéricas
    with tabs[4]:
        st.subheader("📊 Ítem 5 — Distribución de Variables Numéricas")
        st.markdown("Selecciona una variable y ajusta los parámetros para explorar su distribución.")

        num_cols = analyzer.classify_variables()["numeric"]
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_num = st.selectbox("Variable numérica", num_cols, key="num_dist")
        with col2:
            bins = st.slider("Número de bins", 10, 100, 30, key="bins_dist")
        with col3:
            show_hue = st.checkbox("Separar por variable `y`", value=True, key="hue_dist")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Histograma**")
            fig = analyzer.plot_histogram(sel_num, bins=bins, hue="y" if show_hue else None)
            st.pyplot(fig)
        with col_b:
            st.markdown("**KDE (Densidad)**")
            fig = analyzer.plot_kde(sel_num)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📋 Estadísticas de la variable seleccionada")
        stats = df_loaded[sel_num].describe()
        sk    = df_loaded[sel_num].skew()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Media",    f"{stats['mean']:.2f}")
        c2.metric("Mediana",  f"{stats['50%']:.2f}")
        c3.metric("Std Dev",  f"{stats['std']:.2f}")
        c4.metric("Mín",      f"{stats['min']:.2f}")
        c5.metric("Asimetría",f"{sk:.3f}")

        st.markdown(f"""
        **💬 Interpretación:** La variable `{sel_num}` presenta una asimetría de
        **{sk:.3f}** ({'positiva (cola derecha)' if sk > 0 else 'negativa (cola izquierda)'}).
        {'Esto sugiere presencia de valores atípicos altos.' if sk > 1 else
         'La distribución es relativamente simétrica.' if abs(sk) < 0.5 else
         'Existe moderada asimetría.'}
        """)

    # ── ÍTEM 6: Variables categóricas ────────────
    with tabs[5]:
        st.subheader("📊 Ítem 6 — Análisis de Variables Categóricas")

        cat_cols = [c for c in analyzer.classify_variables()["categorical"] if c != "y"]
        sel_cat  = st.selectbox("Variable categórica", cat_cols, key="cat_sel")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("**Frecuencia por categoría**")
            fig = analyzer.plot_barh(sel_cat)
            st.pyplot(fig)
        with col2:
            st.markdown("**Proporción (%)**")
            fig = analyzer.plot_pie(sel_cat)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("**📋 Tabla de conteos y proporciones**")
        vc   = df_loaded[sel_cat].value_counts().reset_index()
        vc.columns = ["Categoría", "Conteo"]
        vc["Proporción (%)"] = (vc["Conteo"] / len(df_loaded) * 100).round(2)
        st.dataframe(vc, use_container_width=True)

    # ── ÍTEM 7: Bivariado numérico vs categórico ─
    with tabs[6]:
        st.subheader("📊 Ítem 7 — Análisis Bivariado (Numérico × Categórico)")
        st.markdown("""
        Exploramos cómo se distribuyen las variables numéricas según la variable
        objetivo `y` y otras variables categóricas.
        """)

        num_cols = analyzer.classify_variables()["numeric"]
        cat_cols_all = analyzer.classify_variables()["categorical"]

        col1, col2 = st.columns(2)
        with col1:
            sel_n = st.selectbox("Variable numérica", num_cols, key="biv_num")
        with col2:
            sel_c = st.selectbox("Variable categórica", cat_cols_all, key="biv_cat",
                                 index=cat_cols_all.index("y") if "y" in cat_cols_all else 0)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Boxplot: {sel_n} por {sel_c}**")
            fig = analyzer.plot_boxplot(sel_n, sel_c)
            st.pyplot(fig)
        with col_b:
            st.markdown(f"**KDE: {sel_n} por {sel_c}**")
            fig = analyzer.plot_kde(sel_n, sel_c)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📋 Estadísticas por grupo")
        grp = df_loaded.groupby(sel_c)[sel_n].describe().round(2)
        st.dataframe(grp, use_container_width=True)

        # Ejemplos fijos del enunciado
        st.markdown("---")
        st.markdown("#### 🎯 Ejemplos del enunciado: age vs y  |  duration vs y")
        c1, c2 = st.columns(2)
        with c1:
            fig = analyzer.plot_boxplot("age", "y")
            st.pyplot(fig)
        with c2:
            fig = analyzer.plot_kde("duration", "y")
            st.pyplot(fig)

    # ── ÍTEM 8: Bivariado cat × cat ──────────────
    with tabs[7]:
        st.subheader("📊 Ítem 8 — Análisis Bivariado (Categórico × Categórico)")
        st.markdown("""
        Analizamos la relación entre dos variables categóricas mediante tablas
        de contingencia y heatmaps de proporción.
        """)

        cat_cols_all = analyzer.classify_variables()["categorical"]
        col1, col2 = st.columns(2)
        with col1:
            sel_c1 = st.selectbox("Variable 1", cat_cols_all, key="catcat1",
                                  index=cat_cols_all.index("education") if "education" in cat_cols_all else 0)
        with col2:
            sel_c2 = st.selectbox("Variable 2", cat_cols_all, key="catcat2",
                                  index=cat_cols_all.index("y") if "y" in cat_cols_all else 1)

        if sel_c1 != sel_c2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Heatmap de proporciones (%)**")
                fig = analyzer.plot_cat_heatmap(sel_c1, sel_c2)
                st.pyplot(fig)
            with col_b:
                st.markdown("**Tasa de aceptación por categoría**")
                fig = analyzer.plot_cat_vs_target(sel_c1, sel_c2)
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("**📋 Tabla de contingencia (conteos)**")
            ct = pd.crosstab(df_loaded[sel_c1], df_loaded[sel_c2])
            st.dataframe(ct, use_container_width=True)

            # Ejemplos fijos del enunciado
            st.markdown("---")
            st.markdown("#### 🎯 Ejemplos del enunciado: education vs y  |  contact vs y")
            c1, c2 = st.columns(2)
            with c1:
                fig = analyzer.plot_cat_vs_target("education", "y")
                st.pyplot(fig)
            with c2:
                fig = analyzer.plot_cat_vs_target("contact", "y")
                st.pyplot(fig)
        else:
            st.warning("Selecciona dos variables distintas.")

    # ── ÍTEM 9: Análisis dinámico ─────────────────
    with tabs[8]:
        st.subheader("⚙️ Ítem 9 — Análisis Dinámico por Parámetros")
        st.markdown("""
        Configura tu propio análisis seleccionando las columnas y filtros
        que deseas explorar.
        """)

        num_cols = analyzer.classify_variables()["numeric"]
        cat_cols = [c for c in analyzer.classify_variables()["categorical"] if c != "y"]

        with st.expander("🎛️ Filtros globales", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_range = st.slider("Rango de edad", int(df_loaded["age"].min()),
                                      int(df_loaded["age"].max()),
                                      (25, 60), key="age_slider")
            with col2:
                dur_range = st.slider("Duración del contacto (s)",
                                      int(df_loaded["duration"].min()),
                                      int(df_loaded["duration"].max()),
                                      (0, 1500), key="dur_slider")
            with col3:
                jobs_sel = st.multiselect("Tipo de trabajo",
                                          sorted(df_loaded["job"].unique()),
                                          default=sorted(df_loaded["job"].unique()),
                                          key="job_ms")

        df_filt = df_loaded[
            (df_loaded["age"].between(*age_range)) &
            (df_loaded["duration"].between(*dur_range)) &
            (df_loaded["job"].isin(jobs_sel))
        ]
        st.info(f"📊 Registros filtrados: **{len(df_filt):,}** de {len(df_loaded):,} "
                f"({len(df_filt)/len(df_loaded)*100:.1f}%)")

        st.markdown("---")
        st.markdown("#### 🔢 Análisis numérico personalizado")
        col1, col2 = st.columns(2)
        with col1:
            x_sel = st.selectbox("Eje X (numérico)", num_cols, key="dyn_x")
        with col2:
            y_sel = st.selectbox("Eje Y (numérico)", num_cols,
                                 index=1 if len(num_cols) > 1 else 0, key="dyn_y")

        show_scatter = st.checkbox("Mostrar scatter plot", value=True, key="show_scatter")
        if show_scatter and x_sel != y_sel:
            an_filt = DataAnalyzer(df_filt)
            fig = an_filt.plot_scatter(x_sel, y_sel)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 🔤 Análisis categórico personalizado")
        cat_dyn = st.selectbox("Variable categórica a explorar", cat_cols, key="dyn_cat")
        an_filt2 = DataAnalyzer(df_filt)
        col_a, col_b = st.columns(2)
        with col_a:
            fig = an_filt2.plot_barh(cat_dyn)
            st.pyplot(fig)
        with col_b:
            fig = an_filt2.plot_cat_vs_target(cat_dyn)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📋 Estadísticas del subconjunto filtrado")
        multi_cols = st.multiselect("Columnas numéricas a describir", num_cols,
                                    default=num_cols[:4], key="desc_multi")
        if multi_cols:
            st.dataframe(df_filt[multi_cols].describe().round(2), use_container_width=True)

    # ── ÍTEM 10: Hallazgos clave ──────────────────
    with tabs[9]:
        st.subheader("💡 Ítem 10 — Hallazgos Clave del EDA")
        st.markdown("Resumen visual e insights principales derivados del análisis exploratorio.")

        # Métricas de alto nivel
        total   = len(df_loaded)
        yes_pct = (df_loaded["y"] == "yes").mean() * 100
        med_dur = df_loaded[df_loaded["y"] == "yes"]["duration"].median()
        top_job = df_loaded[df_loaded["y"] == "yes"]["job"].value_counts().index[0]
        top_month = df_loaded[df_loaded["y"] == "yes"]["month"].value_counts().index[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tasa de conversión", f"{yes_pct:.1f}%", delta="-4pp vs período anterior")
        c2.metric("Duración media (clientes 'yes')", f"{med_dur:.0f} s")
        c3.metric("Trabajo más frecuente (yes)", top_job)
        c4.metric("Mes con más conversiones", top_month.upper())

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🗓️ Tasa de aceptación por mes**")
            month_order = ["jan","feb","mar","apr","may","jun",
                           "jul","aug","sep","oct","nov","dec"]
            mo = df_loaded.groupby("month")["y"].apply(
                lambda x: (x=="yes").mean()*100).reindex(
                [m for m in month_order if m in df_loaded["month"].unique()])
            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.bar(mo.index, mo.values, color=PALETTE[0], edgecolor="white")
            ax.set_title("Tasa de conversión (%) por mes", fontweight="bold")
            ax.set_ylabel("% yes")
            ax.axhline(yes_pct, color="red", linestyle="--", linewidth=1, label=f"Media: {yes_pct:.1f}%")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**📞 Tasa de aceptación por canal de contacto**")
            fig = analyzer.plot_cat_vs_target("contact")
            st.pyplot(fig)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**🎓 Tasa de aceptación por educación**")
            fig = analyzer.plot_cat_vs_target("education")
            st.pyplot(fig)

        with col4:
            st.markdown("**🔥 Correlación entre variables numéricas**")
            fig = analyzer.plot_correlation()
            st.pyplot(fig)


# ══════════════════════════════════════════════
#  MÓDULO 4 — CONCLUSIONES
# ══════════════════════════════════════════════
elif menu == "💡 Conclusiones":
    st.title("💡 Conclusiones Finales")
    st.markdown("Basadas en el Análisis Exploratorio de Datos realizado sobre el dataset BankMarketing.")
    st.markdown("---")

    conclusiones = [
        {
            "num": "1",
            "titulo": "La duración del contacto es el factor más influyente en la conversión",
            "texto": (
                "Los clientes que aceptaron el depósito a plazo (`y = yes`) presentan una "
                "duración mediana de contacto significativamente mayor (~550 s) frente a quienes "
                "rechazaron (~145 s). Esto sugiere que un mayor tiempo de conversación está "
                "fuertemente asociado con la intención de compra. **Recomendación:** capacitar a "
                "los ejecutivos para extender la calidad y profundidad del diálogo con el cliente."
            ),
            "icon": "⏱️",
        },
        {
            "num": "2",
            "titulo": "El canal de contacto celular es más efectivo que el telefónico",
            "texto": (
                "Los clientes contactados por `cellular` presentan una tasa de conversión "
                "superior (~14%) frente a los contactados por `telephone` (~5%). Esto indica "
                "que el canal impacta directamente en la efectividad de la campaña. "
                "**Recomendación:** priorizar el uso del canal celular en futuras campañas y "
                "reducir la inversión en contacto telefónico fijo."
            ),
            "icon": "📱",
        },
        {
            "num": "3",
            "titulo": "Los meses de marzo, septiembre, octubre y diciembre son los más efectivos",
            "texto": (
                "El análisis por mes revela que la tasa de conversión es notablemente superior "
                "en ciertos meses, superando el 40-50% en `mar`, `sep`, `oct` y `dec`, mientras "
                "que en `may` (el mes con más contactos) la tasa cae a ~6%. "
                "**Recomendación:** concentrar los esfuerzos de campaña en los meses de alta "
                "conversión y reducir el volumen de contactos en períodos de baja efectividad."
            ),
            "icon": "📅",
        },
        {
            "num": "4",
            "titulo": "Los clientes con mayor nivel educativo y sin créditos en mora tienen mayor propensión",
            "texto": (
                "Los clientes con educación universitaria (`university.degree`) muestran tasas "
                "de conversión más altas. Asimismo, quienes no presentan mora (`default = no`) "
                "son considerablemente más receptivos. El perfil demográfico óptimo corresponde "
                "a adultos de entre 30 y 60 años, con empleo estable (admin., technician) y sin "
                "deudas en mora. **Recomendación:** segmentar la base de datos y priorizar "
                "este perfil de cliente."
            ),
            "icon": "🎓",
        },
        {
            "num": "5",
            "titulo": "El número excesivo de contactos reduce la probabilidad de conversión",
            "texto": (
                "La variable `campaign` muestra que la mayoría de conversiones ocurren en "
                "los primeros 1-3 contactos. A medida que aumenta el número de llamadas, "
                "la tasa de conversión cae drásticamente. Esto indica que el hostigamiento "
                "al cliente genera rechazo. **Recomendación:** establecer un límite máximo de "
                "3 intentos por cliente y redirigir el esfuerzo hacia nuevos prospectos si no "
                "hay respuesta positiva."
            ),
            "icon": "📊",
        },
    ]

    for c in conclusiones:
        with st.expander(f"{c['icon']} Conclusión {c['num']}: {c['titulo']}", expanded=True):
            st.markdown(c["texto"])

    st.markdown("---")
    st.subheader("🎯 Recomendación Estratégica Global")
    st.success("""
    Para revertir la caída de efectividad del 12% al 8%, el banco debería:

    1. **Segmentar** la base de datos priorizando clientes con perfil de alta propensión
       (educación universitaria, sin mora, contacto celular).
    2. **Concentrar la campaña** en los meses de mayor conversión (mar, sep, oct, dic).
    3. **Limitar los contactos** a un máximo de 3 intentos por cliente.
    4. **Capacitar a ejecutivos** para extender la duración de las conversaciones de calidad.
    5. **Migrar el canal** de telefónico fijo a celular como medio preferente.

    Aplicando estas medidas de forma combinada, se estima factible recuperar
    la efectividad por encima del 10% en la próxima campaña.
    """)

    st.markdown("---")
    st.caption("📁 Proyecto desarrollado como Caso de Estudio N°1 — Especialización Python for Analytics · DMC Institute · 2026")
