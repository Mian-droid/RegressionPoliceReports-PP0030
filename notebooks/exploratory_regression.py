import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "processed"
DENUNCIAS_F = DATA_DIR / "DATASET_Denuncias_Policiales_Enero 2018 a Agosto 2025.csv"
EJEC_F = DATA_DIR / "DATASET_Ejecu_Presup_PP0030_Ene 2019 a Ago 2025.csv"

# Ensure processed directory exists
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def read_denuncias(path):
    # file uses standard CSV with header: ANIO,MES,DPTO_HECHO_NEW,...,cantidad
    # Try different encodings
    encodings = ['utf-8-sig', 'latin1', 'cp1252', 'utf-8', 'iso-8859-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
            print(f"  ✓ Archivo leído con encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"No se pudo leer el archivo con ninguna codificación probada: {encodings}")
    
    # keep relevant columns
    cols = [c for c in df.columns]
    # normalize column names
    df.columns = [c.strip().upper() for c in cols]
    # required
    for c in ["ANIO", "MES", "DPTO_HECHO_NEW", "CANTIDAD"]:
        if c not in df.columns:
            raise ValueError(f"Expected column {c} in denuncias file, got {df.columns.tolist()}")

    # coerce
    df = df[["ANIO", "MES", "DPTO_HECHO_NEW", "CANTIDAD"]].copy()
    df = df.rename(columns={"DPTO_HECHO_NEW": "DEPARTAMENTO"})
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").astype("Int64")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ANIO", "MES"])  # drop malformed rows
    # aggregate counts
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"].str.replace(r"[^0-9\-\.]", "", regex=True), errors="coerce").fillna(0).astype(int)
    df = df.groupby(["ANIO", "MES", "DEPARTAMENTO"], as_index=False)["CANTIDAD"].sum()
    # make period
    df = df[df["MES"] != 0]  # drop month==0 (treated as missing)
    df["period"] = pd.to_datetime(df["ANIO"].astype(int).astype(str) + "-" + df["MES"].astype(int).astype(str).str.zfill(2) + "-01")
    return df[["DEPARTAMENTO", "period", "CANTIDAD"]]


def clean_denuncias(df):
    """
    Limpia el dataset de denuncias policiales:
    - Elimina duplicados exactos
    - Filtra departamentos válidos
    - Elimina outliers extremos en cantidad
    - Normaliza nombres de departamentos
    - Filtra rango de fechas razonable
    """
    print("\n--- LIMPIEZA DE DENUNCIAS ---")
    initial_rows = len(df)
    print(f"Filas iniciales: {initial_rows:,}")
    
    # 1. Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["DEPARTAMENTO", "period", "CANTIDAD"])
    print(f"Duplicados eliminados: {initial_rows - len(df):,}")
    
    # 2. Normalizar nombres de departamentos (capitalizar, trim)
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.strip().str.upper()
    
    # 3. Filtrar departamentos vacíos o inválidos
    before = len(df)
    df = df[df["DEPARTAMENTO"].notna() & (df["DEPARTAMENTO"] != "")]
    df = df[df["DEPARTAMENTO"].str.len() > 2]  # nombre mínimo razonable
    print(f"Departamentos inválidos eliminados: {before - len(df):,}")
    
    # 4. Filtrar fechas razonables (2018-2025)
    before = len(df)
    df = df[(df["period"] >= "2018-01-01") & (df["period"] <= "2025-12-31")]
    print(f"Fechas fuera de rango eliminadas: {before - len(df):,}")
    
    # 5. Filtrar cantidades negativas (no tiene sentido)
    before = len(df)
    df = df[df["CANTIDAD"] >= 0]
    print(f"Cantidades negativas eliminadas: {before - len(df):,}")
    
    # 6. Detectar y eliminar outliers extremos (más de 5 desviaciones estándar)
    # agrupados por departamento
    before = len(df)
    df = df.copy()
    df["z_score"] = df.groupby("DEPARTAMENTO")["CANTIDAD"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    df = df[df["z_score"].abs() <= 5]
    df = df.drop(columns=["z_score"])
    print(f"Outliers extremos eliminados (>5 std): {before - len(df):,}")
    
    # 7. Reagregar por si quedaron duplicados después de normalización
    df = df.groupby(["DEPARTAMENTO", "period"], as_index=False)["CANTIDAD"].sum()
    
    print(f"Filas finales después de limpieza: {len(df):,}")
    print(f"Departamentos únicos: {df['DEPARTAMENTO'].nunique()}")
    print(f"Rango temporal: {df['period'].min().date()} a {df['period'].max().date()}")
    
    return df


def clean_ejecucion(df):
    """
    Limpia el dataset de ejecución presupuestal:
    - Elimina duplicados
    - Filtra departamentos válidos
    - Maneja montos negativos (reversiones)
    - Elimina outliers extremos
    - Normaliza nombres de departamentos
    """
    print("\n--- LIMPIEZA DE EJECUCIÓN PRESUPUESTAL ---")
    initial_rows = len(df)
    print(f"Filas iniciales: {initial_rows:,}")
    
    # 1. Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["DEPARTAMENTO", "period", "MONTO_DEVENGADO"])
    print(f"Duplicados eliminados: {initial_rows - len(df):,}")
    
    # 2. Normalizar departamentos
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.strip().str.upper()
    
    # 3. Filtrar departamentos inválidos
    before = len(df)
    df = df[df["DEPARTAMENTO"].notna() & (df["DEPARTAMENTO"] != "")]
    df = df[df["DEPARTAMENTO"].str.len() > 2]
    print(f"Departamentos inválidos eliminados: {before - len(df):,}")
    
    # 4. Filtrar fechas razonables (2019-2025, según nombre del archivo)
    before = len(df)
    df = df[(df["period"] >= "2019-01-01") & (df["period"] <= "2025-12-31")]
    print(f"Fechas fuera de rango eliminadas: {before - len(df):,}")
    
    # 5. Análisis de montos negativos (son reversiones/correcciones, no errores)
    neg_count = (df["MONTO_DEVENGADO"] < 0).sum()
    print(f"Montos negativos encontrados: {neg_count:,} (reversiones presupuestales, se mantienen)")
    
    # 6. Detectar outliers extremos por departamento
    before = len(df)
    df = df.copy()
    df["z_score"] = df.groupby("DEPARTAMENTO")["MONTO_DEVENGADO"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    df = df[df["z_score"].abs() <= 5]
    df = df.drop(columns=["z_score"])
    print(f"Outliers extremos eliminados (>5 std): {before - len(df):,}")
    
    # 7. Reagregar por si quedaron duplicados
    df = df.groupby(["DEPARTAMENTO", "period"], as_index=False)["MONTO_DEVENGADO"].sum()
    
    print(f"Filas finales después de limpieza: {len(df):,}")
    print(f"Departamentos únicos: {df['DEPARTAMENTO'].nunique()}")
    print(f"Rango temporal: {df['period'].min().date()} a {df['period'].max().date()}")
    print(f"Monto total devengado: S/ {df['MONTO_DEVENGADO'].sum():,.2f}")
    
    return df


def read_ejecucion(path):
    # header: ANO_EJE,MES_EJE,UBIGEO,DEPTO_EJEC_NOMBRE_NEW,...,MONTO_DEVENGADO
    # Try different encodings (utf-8, latin1, utf-8-sig, cp1252)
    encodings = ['utf-8-sig', 'latin1', 'cp1252', 'utf-8', 'iso-8859-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
            print(f"  ✓ Archivo leído con encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"No se pudo leer el archivo con ninguna codificación probada: {encodings}")
    # normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    # try variants
    ano_col = next((c for c in df.columns if c.startswith("ANO") or c == "ANO_EJE" or c == "ANO_EJE"), None)
    mes_col = next((c for c in df.columns if c.startswith("MES") and "EJE" in c) or ("MES_EJE" if "MES_EJE" in df.columns else None), None)
    if "ANO_EJE" in df.columns:
        ano_col = "ANO_EJE"
    if "MES_EJE" in df.columns:
        mes_col = "MES_EJE"

    dept_col = next((c for c in df.columns if "DEPTO" in c and "EJEC" in c), None)
    monto_col = next((c for c in df.columns if "MONTO_DEVENGADO" in c or "MONTO" in c and "DEVENGADO" in c), None)

    # fallback if detection failed
    if ano_col is None:
        ano_col = "ANO_EJE" if "ANO_EJE" in df.columns else df.columns[0]
    if mes_col is None:
        mes_col = "MES_EJE" if "MES_EJE" in df.columns else df.columns[1]
    if dept_col is None:
        # try simple name
        dept_col = next((c for c in df.columns if c.startswith("DEPTO") or c.startswith("DEPART")), None)
    if monto_col is None:
        monto_col = next((c for c in df.columns if "DEVENGADO" in c or c.endswith("MONTO") or "MONTO" in c), None)

    if any(x is None for x in [ano_col, mes_col, dept_col, monto_col]):
        raise ValueError(f"Could not find required columns in ejecucion file; found: {df.columns.tolist()}")

    df = df[[ano_col, mes_col, dept_col, monto_col]].copy()
    df = df.rename(columns={ano_col: "ANIO", mes_col: "MES", dept_col: "DEPARTAMENTO", monto_col: "MONTO_DEVENGADO"})
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").astype("Int64")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ANIO", "MES"])  # drop malformed rows
    # remove rows with MES==0 (no month)
    df = df[df["MES"] != 0]
    # clean monto
    df["MONTO_DEVENGADO"] = pd.to_numeric(df["MONTO_DEVENGADO"].str.replace(r"[^0-9\-\.]", "", regex=True), errors="coerce").fillna(0.0)
    # aggregate per month+department
    df = df.groupby(["ANIO", "MES", "DEPARTAMENTO"], as_index=False)["MONTO_DEVENGADO"].sum()
    df["period"] = pd.to_datetime(df["ANIO"].astype(int).astype(str) + "-" + df["MES"].astype(int).astype(str).str.zfill(2) + "-01")
    return df[["DEPARTAMENTO", "period", "MONTO_DEVENGADO"]]


def make_panel(den, ejec):
    # outer join to preserve months with zero events
    df = pd.merge(den, ejec, on=["DEPARTAMENTO", "period"], how="outer")
    # ensure complete monthly index per department from min to max period
    depts = df["DEPARTAMENTO"].dropna().unique()
    out = []
    for d in depts:
        sub = df[df["DEPARTAMENTO"] == d].set_index("period").sort_index()
        idx = pd.date_range(sub.index.min(), sub.index.max(), freq="MS")
        sub = sub.reindex(idx)
        sub.index.name = "period"
        sub["DEPARTAMENTO"] = d
        sub["CANTIDAD"] = sub["CANTIDAD"].fillna(0).astype(int)
        sub["MONTO_DEVENGADO"] = sub["MONTO_DEVENGADO"].fillna(0.0)
        out.append(sub.reset_index())
    panel = pd.concat(out, ignore_index=True)
    return panel


def create_features(df, lags=(1, 2, 3)):
    df = df.copy()
    df = df.sort_values(["DEPARTAMENTO", "period"]).reset_index(drop=True)
    for lag in lags:
        df[f"MONTO_LAG_{lag}"] = df.groupby("DEPARTAMENTO")["MONTO_DEVENGADO"].shift(lag).fillna(0.0)
    # time features
    df["month"] = df["period"].dt.month
    df["year"] = df["period"].dt.year
    # target
    df["y"] = np.log1p(df["CANTIDAD"].astype(float))
    return df


def train_and_evaluate(df):
    # simple temporal split: train on all but last 6 months (global last date)
    last_period = df["period"].max()
    test_start = (last_period - pd.DateOffset(months=6)) + pd.DateOffset(days=1)
    train = df[df["period"] < test_start].copy()
    test = df[df["period"] >= test_start].copy()

    features = [c for c in df.columns if c.startswith("MONTO_LAG_")] + ["month", "year"]
    X_train = train[features].fillna(0.0)
    y_train = train["y"]
    X_test = test[features].fillna(0.0)
    y_test = test["y"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Test metrics (last 6 months):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # show coefficients for lags
    ridge = pipeline.named_steps["ridge"]
    coef = ridge.coef_
    print("Feature coefficients:")
    for f, c in zip(features, coef):
        print(f"  {f}: {c:.6f}")

    # save model
    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "ridge_baseline.joblib")
    print(f"Saved model to {out_dir / 'ridge_baseline.joblib'}")


def save_clean_data(den, ejec):
    """Guarda los datos limpios para reutilización"""
    print("\n--- GUARDANDO DATOS LIMPIOS ---")
    
    den_clean_path = CLEAN_DIR / "denuncias_clean.csv"
    ejec_clean_path = CLEAN_DIR / "ejecucion_clean.csv"
    
    den.to_csv(den_clean_path, index=False, encoding='utf-8')
    print(f"✓ Denuncias limpias guardadas: {den_clean_path}")
    
    ejec.to_csv(ejec_clean_path, index=False, encoding='utf-8')
    print(f"✓ Ejecución limpia guardada: {ejec_clean_path}")
    
    return den_clean_path, ejec_clean_path


def generate_cleaning_report(den_raw, ejec_raw, den_clean, ejec_clean):
    """Genera reporte de limpieza con estadísticas"""
    print("\n" + "="*60)
    print("REPORTE DE LIMPIEZA DE DATOS")
    print("="*60)
    
    print("\n📊 DENUNCIAS POLICIALES:")
    print(f"  Registros originales:    {len(den_raw):>10,}")
    print(f"  Registros limpios:       {len(den_clean):>10,}")
    print(f"  Eliminados:              {len(den_raw) - len(den_clean):>10,} ({(1 - len(den_clean)/len(den_raw))*100:.1f}%)")
    print(f"  Departamentos únicos:    {den_clean['DEPARTAMENTO'].nunique():>10}")
    print(f"  Periodo:                 {den_clean['period'].min().date()} a {den_clean['period'].max().date()}")
    print(f"  Total denuncias:         {den_clean['CANTIDAD'].sum():>10,}")
    
    print("\n💰 EJECUCIÓN PRESUPUESTAL:")
    print(f"  Registros originales:    {len(ejec_raw):>10,}")
    print(f"  Registros limpios:       {len(ejec_clean):>10,}")
    print(f"  Eliminados:              {len(ejec_raw) - len(ejec_clean):>10,} ({(1 - len(ejec_clean)/len(ejec_raw))*100:.1f}%)")
    print(f"  Departamentos únicos:    {ejec_clean['DEPARTAMENTO'].nunique():>10}")
    print(f"  Periodo:                 {ejec_clean['period'].min().date()} a {ejec_clean['period'].max().date()}")
    print(f"  Monto total devengado:   S/ {ejec_clean['MONTO_DEVENGADO'].sum():>15,.2f}")
    
    # Departamentos en común
    depts_den = set(den_clean['DEPARTAMENTO'].unique())
    depts_ejec = set(ejec_clean['DEPARTAMENTO'].unique())
    common = depts_den & depts_ejec
    only_den = depts_den - depts_ejec
    only_ejec = depts_ejec - depts_den
    
    print("\n🗺️  DEPARTAMENTOS:")
    print(f"  En común:                {len(common):>10}")
    print(f"  Solo en denuncias:       {len(only_den):>10}")
    print(f"  Solo en ejecución:       {len(only_ejec):>10}")
    
    if only_den:
        print(f"  Departamentos solo en denuncias: {', '.join(sorted(only_den))}")
    if only_ejec:
        print(f"  Departamentos solo en ejecución: {', '.join(sorted(only_ejec))}")
    
    print("="*60 + "\n")


def main():
    print("="*60)
    print("PIPELINE DE REGRESIÓN: DENUNCIAS vs EJECUCIÓN PRESUPUESTAL")
    print("="*60)
    
    # 1. LECTURA DE DATOS RAW
    print("\n📖 PASO 1: LECTURA DE DATOS RAW")
    print("-" * 60)
    den_raw = read_denuncias(DENUNCIAS_F)
    print(f"✓ Denuncias cargadas: {len(den_raw):,} registros")
    
    ejec_raw = read_ejecucion(EJEC_F)
    print(f"✓ Ejecución cargada: {len(ejec_raw):,} registros")
    
    # 2. LIMPIEZA DE DATOS
    print("\n🧹 PASO 2: LIMPIEZA DE DATOS")
    print("-" * 60)
    den_clean = clean_denuncias(den_raw)
    ejec_clean = clean_ejecucion(ejec_raw)
    
    # 3. REPORTE DE LIMPIEZA
    generate_cleaning_report(den_raw, ejec_raw, den_clean, ejec_clean)
    
    # 4. GUARDAR DATOS LIMPIOS
    save_clean_data(den_clean, ejec_clean)
    
    # 5. CREACIÓN DE PANEL
    print("\n🔗 PASO 3: CREACIÓN DE PANEL TEMPORAL")
    print("-" * 60)
    panel = make_panel(den_clean, ejec_clean)
    print(f"✓ Panel creado: {len(panel):,} registros")
    print(f"✓ Departamentos: {panel['DEPARTAMENTO'].nunique()}")
    print(f"✓ Periodo: {panel['period'].min().date()} a {panel['period'].max().date()}")
    
    # 6. FEATURE ENGINEERING
    print("\n⚙️  PASO 4: INGENIERÍA DE CARACTERÍSTICAS")
    print("-" * 60)
    panel = create_features(panel)
    features = [c for c in panel.columns if c.startswith("MONTO_LAG_")]
    print(f"✓ Features creados: {len(features)} lags + 2 temporales (month, year)")
    print(f"✓ Lags: {', '.join(features)}")
    
    # 7. ENTRENAMIENTO Y EVALUACIÓN
    print("\n🤖 PASO 5: ENTRENAMIENTO Y EVALUACIÓN DE MODELO")
    print("-" * 60)
    train_and_evaluate(panel)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)


if __name__ == '__main__':
    main()
