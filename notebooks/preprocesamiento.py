"""
===================================================================================
MÓDULO: PREPROCESAMIENTO DE DATOS (Data Preprocessing)
===================================================================================

Responsabilidades:
- Limpieza de datos (duplicados, valores inválidos, outliers)
- Alineación temporal de datasets (2019-2025)
- Normalización de departamentos
- Validación de calidad
- Generación de reportes de limpieza con tablas formateadas

Output: DataFrames limpios y validados listos para crear el panel
===================================================================================
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from typing import Dict, Any


def clean_denuncias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de denuncias policiales:
    - Elimina duplicados
    - Filtra departamentos válidos
    - Alineación temporal: descarta datos < 2019-01-01
    - Elimina valores negativos
    - Detecta y elimina outliers extremos (>5σ)
    
    Args:
        df: DataFrame con columnas [DEPARTAMENTO, period, CANTIDAD]
        
    Returns:
        DataFrame limpio
    """
    print("\n" + "="*80)
    print(" "*22 + "🧹 LIMPIEZA: DENUNCIAS POLICIALES")
    print("="*80)
    
    initial_rows = len(df)
    steps = []
    current_rows = initial_rows
    
    # 1. Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["DEPARTAMENTO", "period", "CANTIDAD"])
    eliminated = current_rows - len(df)
    steps.append(["1. Duplicados exactos", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 2. Normalizar departamentos
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.strip().str.upper()
    
    # 3. Filtrar departamentos inválidos
    df = df[df["DEPARTAMENTO"].notna() & (df["DEPARTAMENTO"] != "")]
    df = df[df["DEPARTAMENTO"].str.len() > 2]
    eliminated = current_rows - len(df)
    steps.append(["2. Departamentos inválidos", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 4. ⚠️ ALINEACIÓN TEMPORAL: Filtrar datos < 2019-01-01
    # Justificación: Dataset de ejecución presupuestal inicia en 2019
    df = df[df["period"] >= "2019-01-01"]
    eliminated = current_rows - len(df)
    steps.append(["3. Alineación temporal (2019+)", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 5. Eliminar cantidades negativas (errores de captura)
    df = df[df["CANTIDAD"] >= 0]
    eliminated = current_rows - len(df)
    steps.append(["4. Cantidades negativas", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 6. Detectar outliers extremos por departamento (Z-score > 5)
    df = df.copy()
    df["z_score"] = df.groupby("DEPARTAMENTO")["CANTIDAD"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    df = df[df["z_score"].abs() <= 5]
    df = df.drop(columns=["z_score"])
    eliminated = current_rows - len(df)
    steps.append(["5. Outliers (>5σ)", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # Mostrar tabla de pasos
    print(f"\n📊 Proceso de limpieza paso a paso:")
    print(tabulate(steps, 
                   headers=["Paso", "Antes", "Después", "Eliminados", "% Elim."], 
                   tablefmt="fancy_grid"))
    
    # Resumen final
    print(f"\n✅ Resumen:")
    print(f"   • Registros iniciales:      {initial_rows:>10,}")
    print(f"   • Registros finales:        {len(df):>10,}")
    print(f"   • Total eliminados:         {initial_rows - len(df):>10,} ({(1-len(df)/initial_rows)*100:.2f}%)")
    print(f"   • Departamentos únicos:     {df['DEPARTAMENTO'].nunique():>10}")
    print(f"   • Rango temporal:           {df['period'].min().date()} a {df['period'].max().date()}")
    
    return df


def clean_ejecucion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de ejecución presupuestal:
    - Elimina duplicados
    - Filtra departamentos válidos
    - Maneja montos negativos (reversiones)
    - Elimina outliers extremos
    - Normaliza nombres de departamentos
    
    Args:
        df: DataFrame con columnas [DEPARTAMENTO, period, MONTO_DEVENGADO]
        
    Returns:
        DataFrame limpio
    """
    print("\n" + "="*80)
    print(" "*22 + "🧹 LIMPIEZA: EJECUCIÓN PRESUPUESTAL PP0030")
    print("="*80)
    
    initial_rows = len(df)
    steps = []
    current_rows = initial_rows
    
    # 1. Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["DEPARTAMENTO", "period", "MONTO_DEVENGADO"])
    eliminated = current_rows - len(df)
    steps.append(["1. Duplicados exactos", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 2. Normalizar departamentos
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.strip().str.upper()
    
    # 3. Filtrar departamentos inválidos
    before = len(df)
    df = df[df["DEPARTAMENTO"].notna() & (df["DEPARTAMENTO"] != "")]
    df = df[df["DEPARTAMENTO"].str.len() > 2]
    eliminated = current_rows - len(df)
    steps.append(["2. Departamentos inválidos", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 4. Filtrar fechas razonables (2019-2025, según nombre del archivo)
    before = len(df)
    df = df[(df["period"] >= "2019-01-01") & (df["period"] <= "2025-12-31")]
    eliminated = current_rows - len(df)
    steps.append(["3. Fechas fuera de rango", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 5. Análisis de montos negativos (son reversiones/correcciones, no errores)
    neg_count = (df["MONTO_DEVENGADO"] < 0).sum()
    neg_monto = df[df["MONTO_DEVENGADO"] < 0]["MONTO_DEVENGADO"].sum()
    steps.append(["4. Montos negativos", current_rows, len(df), 0, "0.00% (conservados)"])
    
    # 6. Detectar outliers extremos por departamento
    before = len(df)
    df = df.copy()
    df["z_score"] = df.groupby("DEPARTAMENTO")["MONTO_DEVENGADO"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    df = df[df["z_score"].abs() <= 5]
    df = df.drop(columns=["z_score"])
    eliminated = current_rows - len(df)
    steps.append(["5. Outliers (>5σ)", current_rows, len(df), eliminated, f"{eliminated/current_rows*100:.2f}%"])
    current_rows = len(df)
    
    # 7. Reagregar por si quedaron duplicados
    df = df.groupby(["DEPARTAMENTO", "period"], as_index=False)["MONTO_DEVENGADO"].sum()
    
    # Mostrar tabla de pasos
    print(f"\n📊 Proceso de limpieza paso a paso:")
    print(tabulate(steps, 
                   headers=["Paso", "Antes", "Después", "Eliminados", "% Elim."], 
                   tablefmt="fancy_grid"))
    
    # Tabla de análisis de montos negativos
    if neg_count > 0:
        print(f"\n💸 Análisis de Montos Negativos (Reversiones Presupuestales):")
        neg_analysis = [
            ["Registros con montos negativos", f"{neg_count:,}", f"{neg_count/initial_rows*100:.2f}%"],
            ["Total en reversiones", f"S/ {neg_monto:,.2f}", ""],
            ["Acción tomada", "CONSERVADOS", "Son correcciones legítimas"]
        ]
        print(tabulate(neg_analysis, headers=["Descripción", "Valor", "Observación"], tablefmt="fancy_grid"))
    
    # Resumen final
    print(f"\n✅ Resumen:")
    print(f"   • Registros iniciales:      {initial_rows:>10,}")
    print(f"   • Registros finales:        {len(df):>10,}")
    print(f"   • Total eliminados:         {initial_rows - len(df):>10,} ({(1-len(df)/initial_rows)*100:.2f}%)")
    print(f"   • Departamentos únicos:     {df['DEPARTAMENTO'].nunique():>10}")
    print(f"   • Rango temporal:           {df['period'].min().date()} a {df['period'].max().date()}")
    print(f"   • Monto total devengado:    S/ {df['MONTO_DEVENGADO'].sum():>15,.2f}")
    
    return df


def generate_cleaning_report(df_den_raw: pd.DataFrame, df_eje_raw: pd.DataFrame,
                             df_den_clean: pd.DataFrame, df_eje_clean: pd.DataFrame) -> None:
    """
    Genera reporte comparativo de limpieza con tablas formateadas.
    
    Args:
        df_den_raw: DataFrame de denuncias antes de limpieza
        df_eje_raw: DataFrame de ejecución antes de limpieza
        df_den_clean: DataFrame de denuncias después de limpieza
        df_eje_clean: DataFrame de ejecución después de limpieza
    """
    print("\n" + "="*80)
    print(" "*25 + "📋 REPORTE DE CALIDAD DE DATOS")
    print("="*80)
    
    # Tabla 1: Resumen general
    print("\n📊 1. RESUMEN GENERAL DE LIMPIEZA")
    summary_data = [
        ["Denuncias Policiales", 
         f"{len(df_den_raw):,}", 
         f"{len(df_den_clean):,}", 
         f"{len(df_den_raw) - len(df_den_clean):,}",
         f"{(1 - len(df_den_clean)/len(df_den_raw))*100:.2f}%"],
        ["Ejecución Presupuestal PP0030", 
         f"{len(df_eje_raw):,}", 
         f"{len(df_eje_clean):,}", 
         f"{len(df_eje_raw) - len(df_eje_clean):,}",
         f"{(1 - len(df_eje_clean)/len(df_eje_raw))*100:.2f}%"]
    ]
    print(tabulate(summary_data, 
                   headers=["Dataset", "Filas Iniciales", "Filas Finales", "Eliminadas", "% Reducción"],
                   tablefmt="fancy_grid"))
    
    # Tabla 2: Estadísticas descriptivas
    print("\n📊 2. ESTADÍSTICAS DESCRIPTIVAS (DESPUÉS DE LIMPIEZA)")
    stats_data = [
        ["Denuncias", 
         f"{df_den_clean['CANTIDAD'].mean():.2f}",
         f"{df_den_clean['CANTIDAD'].median():.2f}",
         f"{df_den_clean['CANTIDAD'].std():.2f}",
         f"{df_den_clean['CANTIDAD'].min():.0f}",
         f"{df_den_clean['CANTIDAD'].max():.0f}"],
        ["Monto Devengado (S/)", 
         f"{df_eje_clean['MONTO_DEVENGADO'].mean():,.2f}",
         f"{df_eje_clean['MONTO_DEVENGADO'].median():,.2f}",
         f"{df_eje_clean['MONTO_DEVENGADO'].std():,.2f}",
         f"{df_eje_clean['MONTO_DEVENGADO'].min():,.2f}",
         f"{df_eje_clean['MONTO_DEVENGADO'].max():,.2f}"]
    ]
    print(tabulate(stats_data,
                   headers=["Variable", "Media", "Mediana", "Desv. Est.", "Min", "Max"],
                   tablefmt="fancy_grid"))
    
    # Tabla 3: Cobertura geográfica
    print("\n📊 3. COBERTURA GEOGRÁFICA")
    geo_data = [
        ["Denuncias", 
         f"{df_den_clean['DEPARTAMENTO'].nunique()}",
         ", ".join(sorted(df_den_clean['DEPARTAMENTO'].unique())[:5]) + "..."],
        ["Ejecución Presupuestal", 
         f"{df_eje_clean['DEPARTAMENTO'].nunique()}",
         ", ".join(sorted(df_eje_clean['DEPARTAMENTO'].unique())[:5]) + "..."]
    ]
    print(tabulate(geo_data,
                   headers=["Dataset", "Departamentos", "Ejemplos"],
                   tablefmt="fancy_grid"))
    
    # Tabla 4: Alineación temporal
    print("\n📊 4. ALINEACIÓN TEMPORAL")
    temporal_data = [
        ["Denuncias (Original)", 
         str(df_den_raw['period'].min().date()),
         str(df_den_raw['period'].max().date()),
         f"{(df_den_raw['period'].max() - df_den_raw['period'].min()).days // 30} meses"],
        ["Denuncias (Limpio)", 
         str(df_den_clean['period'].min().date()),
         str(df_den_clean['period'].max().date()),
         f"{(df_den_clean['period'].max() - df_den_clean['period'].min()).days // 30} meses"],
        ["Ejecución Presupuestal", 
         str(df_eje_clean['period'].min().date()),
         str(df_eje_clean['period'].max().date()),
         f"{(df_eje_clean['period'].max() - df_eje_clean['period'].min()).days // 30} meses"]
    ]
    print(tabulate(temporal_data,
                   headers=["Dataset", "Inicio", "Fin", "Duración"],
                   tablefmt="fancy_grid"))
    
    print("\n" + "="*80)
    print("✅ Reporte de calidad generado exitosamente")
    print("="*80)


def save_clean_data(df_den: pd.DataFrame, df_eje: pd.DataFrame, 
                    output_dir: str = "../data/processed") -> None:
    """
    Guarda los datasets limpios en formato CSV.
    
    Args:
        df_den: DataFrame de denuncias limpio
        df_eje: DataFrame de ejecución limpio
        output_dir: Directorio de salida
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    den_file = output_path / "denuncias_clean.csv"
    eje_file = output_path / "ejecucion_clean.csv"
    
    df_den.to_csv(den_file, index=False, encoding="utf-8-sig")
    df_eje.to_csv(eje_file, index=False, encoding="utf-8-sig")
    
    print(f"\n💾 Datos limpios guardados:")
    print(f"   • {den_file}")
    print(f"   • {eje_file}")


if __name__ == "__main__":
    # Prueba del módulo
    from entrada import load_data
    
    print("🧪 PRUEBA DEL MÓDULO DE PREPROCESAMIENTO")
    
    # Cargar datos
    df_den_raw, df_eje_raw = load_data("../data/raw")
    
    # Limpiar
    df_den_clean = clean_denuncias(df_den_raw.copy())
    df_eje_clean = clean_ejecucion(df_eje_raw.copy())
    
    # Generar reporte
    generate_cleaning_report(df_den_raw, df_eje_raw, df_den_clean, df_eje_clean)
    
    # Guardar
    save_clean_data(df_den_clean, df_eje_clean)
