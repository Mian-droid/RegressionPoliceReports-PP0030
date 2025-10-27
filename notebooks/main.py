"""
===================================================================================
SCRIPT PRINCIPAL: PIPELINE COMPLETO DE REGRESIÓN
===================================================================================

Orquesta todo el pipeline usando los módulos modularizados:
✅ entrada.py: Carga de datos raw
✅ preprocesamiento.py: Limpieza y validación
⏳ PROCESAMIENTO: Creación de panel y features (temporal en este archivo)
⏳ SALIDA: Entrenamiento, evaluación y guardado (temporal en este archivo)

Ejecutar: python notebooks\main.py
===================================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Importar módulos propios
from entrada import load_data
from preprocesamiento import clean_denuncias, clean_ejecucion, generate_cleaning_report, save_clean_data


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PROCESAMIENTO: Creación de Panel y Features
# (TODO: Modularizar en procesamiento.py)
# ==============================================================================

def make_panel(den, ejec):
    """
    Crea panel balanceado con rango temporal común (2019-2025).
    Alinea ambos datasets al mismo periodo para análisis de regresión.
    
    Args:
        den: DataFrame de denuncias limpio
        ejec: DataFrame de ejecución limpio
        
    Returns:
        DataFrame con panel balanceado
    """
    print("\n" + "="*80)
    print(" "*25 + "📊 CREACIÓN DE PANEL BALANCEADO")
    print("="*80)
    
    # outer join to preserve months with zero events
    df = pd.merge(den, ejec, on=["DEPARTAMENTO", "period"], how="outer")
    
    # Definir rango temporal común: desde 2019-01 (inicio del dataset de ejecución)
    min_period = pd.Timestamp("2019-01-01")
    max_period = df["period"].max()
    
    print(f"   • Rango temporal común: {min_period.date()} a {max_period.date()}")
    
    # Filtrar al rango común
    df = df[(df["period"] >= min_period) & (df["period"] <= max_period)]
    
    # Crear índice completo por departamento
    depts = df["DEPARTAMENTO"].dropna().unique()
    out = []
    for d in depts:
        sub = df[df["DEPARTAMENTO"] == d].set_index("period").sort_index()
        idx = pd.date_range(min_period, max_period, freq="MS")
        sub = sub.reindex(idx)
        sub.index.name = "period"
        sub["DEPARTAMENTO"] = d
        sub["CANTIDAD"] = sub["CANTIDAD"].fillna(0).astype(int)
        sub["MONTO_DEVENGADO"] = sub["MONTO_DEVENGADO"].fillna(0.0)
        out.append(sub.reset_index())
    
    panel = pd.concat(out, ignore_index=True)
    
    print(f"   • Panel shape: {panel.shape}")
    print(f"   • Departamentos: {panel['DEPARTAMENTO'].nunique()}")
    print(f"   • Periodos: {panel['period'].nunique()}")
    print(f"   • Total registros: {len(panel):,}")
    
    return panel


def create_features(df, lags=(1, 2, 3)):
    """
    Crea features para el modelo de regresión:
    - Lags de MONTO_DEVENGADO (1, 2, 3 meses)
    - Features temporales (mes, año)
    - Target: log1p(CANTIDAD)
    
    Args:
        df: DataFrame con panel balanceado
        lags: Tupla con los lags a crear
        
    Returns:
        DataFrame con features creados
    """
    print("\n" + "="*80)
    print(" "*28 + "🔧 INGENIERÍA DE FEATURES")
    print("="*80)
    
    df = df.copy()
    df = df.sort_values(["DEPARTAMENTO", "period"]).reset_index(drop=True)
    
    # Crear lags de MONTO_DEVENGADO
    for lag in lags:
        df[f"MONTO_LAG_{lag}"] = df.groupby("DEPARTAMENTO")["MONTO_DEVENGADO"].shift(lag).fillna(0.0)
    
    # Features temporales
    df["month"] = df["period"].dt.month
    df["year"] = df["period"].dt.year
    
    # Target (transformación log para estabilizar varianza)
    df["y"] = np.log1p(df["CANTIDAD"].astype(float))
    
    print(f"   • Features creados:")
    print(f"      - Lags: MONTO_LAG_{list(lags)}")
    print(f"      - Temporales: month, year")
    print(f"      - Target: y = log1p(CANTIDAD)")
    print(f"   • Shape final: {df.shape}")
    print(f"   • Valores nulos: {df.isnull().sum().sum()}")
    
    return df


# ==============================================================================
# SALIDA: Entrenamiento y Evaluación
# (TODO: Modularizar en salida.py)
# ==============================================================================

def train_and_evaluate(df):
    """
    Entrena modelo Ridge con validación temporal.
    
    Args:
        df: DataFrame con features creados
        
    Returns:
        Modelo entrenado (pipeline)
    """
    print("\n" + "="*80)
    print(" "*22 + "🤖 ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
    print("="*80)
    
    # Split temporal: últimos 6 meses como test
    last_period = df["period"].max()
    test_start = (last_period - pd.DateOffset(months=6)) + pd.DateOffset(days=1)
    train = df[df["period"] < test_start].copy()
    test = df[df["period"] >= test_start].copy()
    
    # Features y target
    features = [c for c in df.columns if c.startswith("MONTO_LAG_")] + ["month", "year"]
    X_train = train[features].fillna(0.0)
    y_train = train["y"]
    X_test = test[features].fillna(0.0)
    y_test = test["y"]
    
    # Pipeline: StandardScaler + RidgeCV
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    preds_train = pipeline.predict(X_train)
    preds_test = pipeline.predict(X_test)
    
    # Métricas
    rmse_train = mean_squared_error(y_train, preds_train, squared=False)
    mae_train = mean_absolute_error(y_train, preds_train)
    r2_train = r2_score(y_train, preds_train)
    
    rmse_test = mean_squared_error(y_test, preds_test, squared=False)
    mae_test = mean_absolute_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)
    
    # Tabla 1: Métricas de desempeño
    print("\n📊 1. MÉTRICAS DE DESEMPEÑO")
    metrics_data = [
        ["RMSE", f"{rmse_train:.4f}", f"{rmse_test:.4f}"],
        ["MAE", f"{mae_train:.4f}", f"{mae_test:.4f}"],
        ["R² Score", f"{r2_train:.4f}", f"{r2_test:.4f}"]
    ]
    print(tabulate(metrics_data, 
                   headers=["Métrica", "Train", "Test"],
                   tablefmt="fancy_grid"))
    
    # Tabla 2: Coeficientes del modelo
    print("\n📊 2. COEFICIENTES DEL MODELO (Top 10)")
    ridge = pipeline.named_steps["ridge"]
    coefs = pd.DataFrame({
        "Feature": features,
        "Coeficiente": ridge.coef_
    }).sort_values("Coeficiente", key=abs, ascending=False).head(10)
    
    coef_table = []
    for _, row in coefs.iterrows():
        direction = "↑ Positivo" if row["Coeficiente"] > 0 else "↓ Negativo"
        coef_table.append([row["Feature"], f"{row['Coeficiente']:.6f}", direction])
    
    print(tabulate(coef_table,
                   headers=["Feature", "Valor", "Dirección"],
                   tablefmt="fancy_grid"))
    
    # Tabla 3: Información del split temporal
    print("\n📊 3. SPLIT TEMPORAL")
    split_data = [
        ["Train", len(train), str(train["period"].min().date()), str(train["period"].max().date())],
        ["Test", len(test), str(test["period"].min().date()), str(test["period"].max().date())]
    ]
    print(tabulate(split_data,
                   headers=["Dataset", "Registros", "Inicio", "Fin"],
                   tablefmt="fancy_grid"))
    
    print(f"\n✅ Mejor alpha (Ridge): {ridge.alpha_:.4f}")
    
    return pipeline


def save_model(pipeline, model_name="ridge_baseline.joblib"):
    """
    Guarda el modelo entrenado.
    
    Args:
        pipeline: Pipeline entrenado
        model_name: Nombre del archivo
    """
    model_path = MODELS_DIR / model_name
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Ejecuta el pipeline completo:
    1. Carga de datos (entrada.py)
    2. Limpieza (preprocesamiento.py)
    3. Creación de panel y features
    4. Entrenamiento y evaluación
    5. Guardado de resultados
    """
    print("\n" + "="*80)
    print(" "*20 + "🚀 PIPELINE DE REGRESIÓN: DENUNCIAS vs PP0030")
    print("="*80)
    
    # PASO 1: ENTRADA - Cargar datos raw
    print("\n📥 PASO 1/5: CARGA DE DATOS")
    data_dir = ROOT / "data" / "raw"
    df_den_raw, df_eje_raw = load_data(str(data_dir))
    
    # PASO 2: PREPROCESAMIENTO - Limpiar datos
    print("\n🧹 PASO 2/5: LIMPIEZA DE DATOS")
    df_den_clean = clean_denuncias(df_den_raw.copy())
    df_eje_clean = clean_ejecucion(df_eje_raw.copy())
    
    # Generar reporte de limpieza
    generate_cleaning_report(df_den_raw, df_eje_raw, df_den_clean, df_eje_clean)
    
    # Guardar datos limpios
    processed_dir = ROOT / "data" / "processed"
    save_clean_data(df_den_clean, df_eje_clean, str(processed_dir))
    
    # PASO 3: PROCESAMIENTO - Crear panel balanceado
    print("\n📊 PASO 3/5: CREACIÓN DE PANEL Y FEATURES")
    panel = make_panel(df_den_clean, df_eje_clean)
    df_features = create_features(panel)
    
    # PASO 4: SALIDA - Entrenar y evaluar modelo
    print("\n🤖 PASO 4/5: ENTRENAMIENTO Y EVALUACIÓN")
    pipeline = train_and_evaluate(df_features)
    
    # PASO 5: Guardar modelo
    print("\n💾 PASO 5/5: GUARDADO DE RESULTADOS")
    save_model(pipeline)
    
    print("\n" + "="*80)
    print(" "*30 + "✅ PIPELINE COMPLETADO")
    print("="*80)
    print(f"\n📂 Archivos generados:")
    print(f"   • data/processed/denuncias_clean.csv")
    print(f"   • data/processed/ejecucion_clean.csv")
    print(f"   • models/ridge_baseline.joblib")
    print("\n📸 Captura las tablas para tu informe con Windows + Shift + S")
    print("="*80)


if __name__ == "__main__":
    main()
