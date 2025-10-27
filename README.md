
# Modelo de Regresión: Ejecución Presupuestal PP0030 vs Denuncias Policiales

## 📋 Descripción del Proyecto

Este proyecto desarrolla un **modelo de regresión** para analizar la influencia de la **Ejecución Presupuestal del Programa Presupuestal PP0030** en la **variación de la Tasa de Denuncias Policiales** en Perú.

### Objetivo
Determinar si existe una relación estadísticamente significativa entre:
- **Variable predictora:** Monto devengado del presupuesto PP0030 (por departamento y mes)
- **Variable objetivo:** Cantidad de denuncias policiales (por departamento y mes)


## 📊 Datasets

### 1. Denuncias Policiales (2018-2025)
- **Archivo:** `data/raw/DATASET_Denuncias_Policiales_Enero 2018 a Agosto 2025.csv`
- **Granularidad:** Mensual por departamento
- **Columnas clave:** ANIO, MES, DPTO_HECHO_NEW, cantidad

### 2. Ejecución Presupuestal PP0030 (2019-2025)
- **Archivo:** `data/raw/DATASET_Ejecu_Presup_PP0030_Ene 2019 a Ago 2025.csv`
- **Granularidad:** Mensual por departamento
- **Columnas clave:** ANO_EJE, MES_EJE, DEPTO_EJEC_NOMBRE_NEW, MONTO_DEVENGADO


## 🚀 Inicio Rápido

### 1. Configurar Entorno Virtual

# Crear entorno virtual con Python 3.11
py -3.11 -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

### 2. Ejecutar Pipeline de Limpieza

python notebooks/exploratory_regression.py

## 📁 Estructura del Proyecto

```
RegressionPoliceReports-PP0030/
│
├── data/
│   ├── raw/                          # Datos originales (no modificar)
│   │   ├── DATASET_Denuncias_Policiales_Enero 2018 a Agosto 2025.csv
│   │   └── DATASET_Ejecu_Presup_PP0030_Ene 2019 a Ago 2025.csv
│   │
│   └── processed/                    # Datos limpios (generados automáticamente)
│       ├── denuncias_clean.csv
│       └── ejecucion_clean.csv
│
├── notebooks/
│   └── exploratory_regression.py    # Pipeline principal de limpieza y modelado
│
├── models/
│   └── ridge_baseline.joblib        # Modelo entrenado (generado automáticamente)
│
├── requirements.txt                  # Dependencias Python (pip)
├── environment.yml                   # Dependencias Conda (alternativa)
├── INSTALL_WINDOWS.md               # Guía de instalación detallada para Windows
└── README.md                         # Este archivo
```


## 🤖 Modelado

### Features Creados
- **Lags temporales:** MONTO_LAG_1, MONTO_LAG_2, MONTO_LAG_3 (montos devengados en meses anteriores)
- **Features temporales:** month (mes del año), year (año)

### Modelo Baseline
- **Algoritmo:** Ridge Regression con validación cruzada (RidgeCV)
- **Preprocesamiento:** StandardScaler (normalización de features)
- **Validación:** Split temporal (últimos 6 meses como test set)
- **Métricas:** RMSE, MAE, R²
 

## 📈 Próximos Pasos

- [ ] **EDA completo:** Visualizaciones de series temporales, correlaciones, estacionalidad
- [ ] **Modelos adicionales:** Lasso, RandomForest, XGBoost
- [ ] **Validación temporal:** TimeSeriesSplit, walk-forward validation
- [ ] **Pruebas de causalidad:** Granger causality test
- [ ] **Análisis de residuales:** ACF/PACF, pruebas de estacionariedad (ADF)
- [ ] **Notebook interactivo:** Jupyter notebook con visualizaciones y explicaciones
- [ ] **Documentación final:** Reporte ejecutivo con hallazgos y recomendaciones

---

## 👥 Equipo

Este proyecto es desarrollado como parte de un trabajo de análisis de Inteligencia Artificial aplicada a datos gubernamentales.

LUYO DAGA, MIGUEL ANGEL
RODRIGUEZ ALMORA, AMIRA PAOLA
RAYMUNDO MOREYRA, PIERO EDUARDO
ARAGON VILCA, RODRIGO RAYHAN JEREMY
YABAR REAÑO, SAID SANTIAGO

## 📝 Notas

- **Relación causal vs correlación:** Este modelo identifica asociaciones estadísticas, no necesariamente causalidad directa.
- **Datos limpios reutilizables:** Los CSVs procesados en `data/processed/` pueden usarse para otros análisis sin reprocesar.
- **Modelo versionable:** El archivo `.joblib` permite versionar y comparar diferentes iteraciones del modelo.


**Última actualización:** Octubre 2025
