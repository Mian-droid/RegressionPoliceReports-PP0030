"""
===================================================================================
SCRIPT PRINCIPAL: PIPELINE COMPLETO DE REGRESIÓN
===================================================================================

Orquesta todo el pipeline usando los módulos modularizados:
✅ entrada.py: Carga de datos raw
✅ preprocesamiento.py: Limpieza y validación
✅ PROCESAMIENTO: Creación de panel y features
✅ SALIDA: Entrenamiento, evaluación y guardado

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
from procesamiento import evaluar_modelos_cv
from salida import evaluar_modelo_elegido # Importación del módulo de salida

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PROCESAMIENTO: Creación de Panel y Features
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
    
    print(f"   • Rango temporal común: {min_period.date()} a {max_period.date()}")
    
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
    
    print(f"   • Panel shape: {panel.shape}")
    print(f"   • Departamentos: {panel['DEPARTAMENTO'].nunique()}")
    print(f"   • Periodos: {panel['period'].nunique()}")
    print(f"   • Total registros: {len(panel):,}")
    
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
    
    print(f"   • Features creados:")
    print(f"      - Lags: MONTO_LAG_{list(lags)}")
    print(f"      - Temporales: month, year")
    print(f"      - Target: y = log1p(CANTIDAD)")
    print(f"   • Shape final: {df.shape}")
    print(f"   • Valores nulos: {df.isnull().sum().sum()}")
    
    return df


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Ejecuta el pipeline completo.
    """
    print("\n" + "="*80)
    print(" "*20 + "🚀 PIPELINE DE REGRESIÓN: DENUNCIAS vs PP0030")
    print("="*80)
    
    # PASO 1: ENTRADA - Cargar datos raw
    print("\n📥 PASO 1/5: CARGA DE DATOS")
    # Nota: asumiendo que este script se corre desde notebooks/
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
    
    # PASO 3: PROCESAMIENTO - Crear panel balanceado y features
    print("\n📊 PASO 3/5: CREACIÓN DE PANEL Y FEATURES")
    panel = make_panel(df_den_clean, df_eje_clean)
    df_features = create_features(panel)
    
    # PASO 4: MODELADO - Evaluación de modelos con CV
    print("\n\n🤖 PASO 4/5: EVALUACIÓN DE MODELOS (K-FOLD CV)")
    resultados = evaluar_modelos_cv(df_features)

    print("\n📊 Resultados de CV promedio por modelo:")
    cv_df = resultados["cv_results"]
    # Mostrar tabla resumida
    print(tabulate(cv_df[['model', 'mse_mean', 'mae_mean', 'r2_mean']].values, 
                   headers=["Modelo", "MSE (Mean)", "MAE (Mean)", "R² (Mean)"], 
                   tablefmt="fancy_grid", 
                   floatfmt=".4f"))

    # PASO 5: SALIDA - Evaluación final y diagnóstico gráfico
    print("\n\n📈 PASO 5/5: EVALUACIÓN FINAL DEL MEJOR MODELO")
    evaluar_modelo_elegido(resultados)

    print("\n" + "="*80)
    print(" "*30 + "✅ PIPELINE COMPLETADO")
    print("="*80)
    print(f"\n📂 Archivos generados:")
    print(f"   • data/processed/denuncias_clean.csv")
    print(f"   • data/processed/ejecucion_clean.csv")
    print("\n📸 No olvides capturar todas las tablas y gráficos para tu informe final.")
    print("="*80)


if __name__ == "__main__":
    main()