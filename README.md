# 🏦 BankMarketing EDA — Análisis Exploratorio de Datos

> **Caso de Estudio N°1** · Especialización en Python for Analytics · DMC Institute

---

## 📋 Descripción del Proyecto

Aplicación interactiva construida con **Streamlit** para el Análisis Exploratorio de Datos (EDA) del dataset `BankMarketing.csv`, correspondiente a una campaña de marketing telefónico de una institución financiera portuguesa.

El objetivo es descubrir **patrones, relaciones y comportamientos** en los datos para apoyar la toma de decisiones comerciales frente a una caída en la efectividad de campaña (del 12% al 8%).

---

## 🚀 Demo en vivo

🔗 [Ver aplicación desplegada en Streamlit Cloud](#) *(reemplazar con tu link)*

---

## 📊 Estructura de la Aplicación

| Módulo | Descripción |
|--------|-------------|
| 🏠 Home | Presentación del proyecto, autor y dataset |
| 📂 Carga del Dataset | Upload de CSV, vista previa y dimensiones |
| 🔍 Análisis EDA | 10 ítems de análisis con visualizaciones interactivas |
| 💡 Conclusiones | 5 conclusiones y recomendaciones estratégicas |

### 🔍 Ítems del EDA
1. Información general del dataset
2. Clasificación de variables (POO — clase `DataAnalyzer`)
3. Estadísticas descriptivas extendidas
4. Análisis de valores faltantes y `unknown` implícitos
5. Distribución de variables numéricas (histogramas + KDE)
6. Análisis de variables categóricas (barras + pie)
7. Análisis bivariado numérico × categórico (boxplot + KDE)
8. Análisis bivariado categórico × categórico (heatmap)
9. Análisis dinámico con filtros interactivos
10. Hallazgos clave y visualización resumen

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.11**
- **Streamlit** — interfaz web interactiva
- **Pandas** — manipulación de datos
- **NumPy** — cálculos numéricos
- **Matplotlib & Seaborn** — visualizaciones
- **POO** — clase `DataAnalyzer` como núcleo del análisis

---

## ⚙️ Instrucciones de Ejecución Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/bankmarketing-eda.git
cd bankmarketing-eda

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

Luego abre tu navegador en `http://localhost:8501`

---

## 📁 Estructura del Repositorio

```
bankmarketing-eda/
├── app.py               # Aplicación principal Streamlit
├── BankMarketing.csv    # Dataset
├── requirements.txt     # Dependencias
└── README.md            # Este archivo
```

---

## 👤 Autor

**[Tu nombre completo]**
Especialización en Python for Analytics — DMC Institute — 2026

---

## 📄 Dataset

- **Fuente:** UCI Machine Learning Repository — Bank Marketing Dataset
- **Registros:** 41,188
- **Variables:** 21
- **Variable objetivo:** `y` (yes/no — suscripción a depósito a plazo)
