"""
Reporte de Calidad de Datos - Análisis Detallado
Para el proyecto académico: Regresión Denuncias Policiales vs Ejecución Presupuestal

Este script genera un análisis completo de la calidad de los datos ANTES de la limpieza.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DENUNCIAS_F = DATA_DIR / "DATASET_Denuncias_Policiales_Enero 2018 a Agosto 2025.csv"
EJEC_F = DATA_DIR / "DATASET_Ejecu_Presup_PP0030_Ene 2019 a Ago 2025.csv"


def load_raw_data(filepath, name):
    """Carga datos con diferentes encodings"""
    print(f"\n{'='*60}")
    print(f"CARGANDO: {name}")
    print('='*60)
    
    encodings = ['utf-8-sig', 'latin1', 'cp1252', 'utf-8', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"✓ Archivo cargado con encoding: {enc}")
            print(f"✓ Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer el archivo con ningún encoding probado")


def analyze_completeness(df, name):
    """Analiza valores faltantes"""
    print(f"\n--- COMPLETITUD DE DATOS: {name} ---")
    
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    completeness = ((total_cells - missing) / total_cells) * 100
    
    print(f"Total de celdas: {total_cells:,}")
    print(f"Celdas con datos: {total_cells - missing:,}")
    print(f"Celdas vacías: {missing:,}")
    print(f"Completitud: {completeness:.2f}%")
    
    print("\nValores faltantes por columna:")
    missing_cols = df.isnull().sum()
    missing_pct = (missing_cols / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Columna': missing_cols.index,
        'Faltantes': missing_cols.values,
        'Porcentaje': missing_pct.values
    }).sort_values('Faltantes', ascending=False)
    
    print(missing_df[missing_df['Faltantes'] > 0].to_string(index=False))
    
    if missing_df['Faltantes'].sum() == 0:
        print("  ✓ No hay valores faltantes")
    
    return missing_df


def analyze_duplicates(df, name, key_columns=None):
    """Analiza duplicados exactos y por clave"""
    print(f"\n--- DUPLICADOS: {name} ---")
    
    # Duplicados exactos (todas las columnas)
    exact_dupes = df.duplicated().sum()
    print(f"Duplicados exactos (todas columnas): {exact_dupes:,} ({exact_dupes/len(df)*100:.2f}%)")
    
    # Duplicados por clave
    if key_columns:
        key_dupes = df.duplicated(subset=key_columns).sum()
        print(f"Duplicados por clave {key_columns}: {key_dupes:,} ({key_dupes/len(df)*100:.2f}%)")
    
    return exact_dupes, key_dupes if key_columns else 0


def analyze_data_types(df, name):
    """Analiza tipos de datos y detecta inconsistencias"""
    print(f"\n--- TIPOS DE DATOS: {name} ---")
    
    dtype_info = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.values,
        'Únicos': [df[col].nunique() for col in df.columns],
        'Ejemplo': [str(df[col].iloc[0])[:50] if len(df) > 0 else '' for col in df.columns]
    })
    
    print(dtype_info.to_string(index=False))
    
    return dtype_info


def analyze_denuncias_specific(df):
    """Análisis específico para dataset de denuncias"""
    print(f"\n{'='*60}")
    print("ANÁLISIS ESPECÍFICO: DENUNCIAS POLICIALES")
    print('='*60)
    
    df_work = df.copy()
    df_work.columns = [c.strip().upper() for c in df_work.columns]
    
    # 1. Análisis temporal
    print("\n--- ANÁLISIS TEMPORAL ---")
    if 'ANIO' in df_work.columns and 'MES' in df_work.columns:
        df_work['ANIO_NUM'] = pd.to_numeric(df_work['ANIO'], errors='coerce')
        df_work['MES_NUM'] = pd.to_numeric(df_work['MES'], errors='coerce')
        
        print(f"Rango de años: {df_work['ANIO_NUM'].min():.0f} - {df_work['ANIO_NUM'].max():.0f}")
        print(f"Meses únicos: {sorted(df_work['MES_NUM'].dropna().unique().astype(int).tolist())}")
        
        # Detectar MES == 0 (problemas)
        mes_zero = (df_work['MES_NUM'] == 0).sum()
        if mes_zero > 0:
            print(f"Registros con MES=0: {mes_zero:,} (deben ser filtrados)")
        
        # Distribución temporal
        print("\nRegistros por año:")
        year_dist = df_work['ANIO_NUM'].value_counts().sort_index()
        for year, count in year_dist.items():
            if not pd.isna(year):
                print(f"  {int(year)}: {count:>8,} registros")
    
    # 2. Análisis de departamentos
    print("\n--- ANÁLISIS GEOGRÁFICO ---")
    if 'DPTO_HECHO_NEW' in df_work.columns:
        dept_col = 'DPTO_HECHO_NEW'
    elif 'DEPARTAMENTO' in df_work.columns:
        dept_col = 'DEPARTAMENTO'
    else:
        dept_col = None
    
    if dept_col:
        depts = df_work[dept_col].dropna().unique()
        print(f"Departamentos únicos: {len(depts)}")
        
        # Detectar departamentos con nombres raros
        invalid = df_work[df_work[dept_col].str.len() < 3][dept_col].value_counts()
        if len(invalid) > 0:
            print(f"Departamentos con nombres sospechosos (< 3 chars): {len(invalid)}")
        
        # Top departamentos
        print("\nTop 10 departamentos con más denuncias:")
        top_depts = df_work[dept_col].value_counts().head(10)
        for dept, count in top_depts.items():
            print(f"  {dept:<20}: {count:>8,}")
    
    # 3. Análisis de cantidad
    print("\n--- ANÁLISIS DE CANTIDAD ---")
    if 'CANTIDAD' in df_work.columns:
        df_work['CANTIDAD_NUM'] = pd.to_numeric(
            df_work['CANTIDAD'].astype(str).str.replace(r'[^0-9\-\.]', '', regex=True), 
            errors='coerce'
        )
        
        print(f"Total denuncias: {df_work['CANTIDAD_NUM'].sum():,.0f}")
        print(f"Promedio por registro: {df_work['CANTIDAD_NUM'].mean():.2f}")
        print(f"Mediana: {df_work['CANTIDAD_NUM'].median():.0f}")
        print(f"Mínimo: {df_work['CANTIDAD_NUM'].min():.0f}")
        print(f"Máximo: {df_work['CANTIDAD_NUM'].max():.0f}")
        
        # Valores negativos
        negativos = (df_work['CANTIDAD_NUM'] < 0).sum()
        if negativos > 0:
            print(f"  Cantidades negativas: {negativos:,} (deben ser eliminadas)")
        
        # Outliers (método IQR)
        Q1 = df_work['CANTIDAD_NUM'].quantile(0.25)
        Q3 = df_work['CANTIDAD_NUM'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_work['CANTIDAD_NUM'] < (Q1 - 3 * IQR)) | 
                   (df_work['CANTIDAD_NUM'] > (Q3 + 3 * IQR))).sum()
        print(f"  Outliers extremos (3*IQR): {outliers:,} ({outliers/len(df)*100:.2f}%)")


def analyze_ejecucion_specific(df):
    """Análisis específico para dataset de ejecución presupuestal"""
    print(f"\n{'='*60}")
    print("ANÁLISIS ESPECÍFICO: EJECUCIÓN PRESUPUESTAL")
    print('='*60)
    
    df_work = df.copy()
    df_work.columns = [c.strip().upper() for c in df_work.columns]
    
    # 1. Análisis temporal
    print("\n--- ANÁLISIS TEMPORAL ---")
    ano_col = 'ANO_EJE' if 'ANO_EJE' in df_work.columns else None
    mes_col = 'MES_EJE' if 'MES_EJE' in df_work.columns else None
    
    if ano_col and mes_col:
        df_work['ANIO_NUM'] = pd.to_numeric(df_work[ano_col], errors='coerce')
        df_work['MES_NUM'] = pd.to_numeric(df_work[mes_col], errors='coerce')
        
        print(f"Rango de años: {df_work['ANIO_NUM'].min():.0f} - {df_work['ANIO_NUM'].max():.0f}")
        print(f"Meses únicos: {sorted(df_work['MES_NUM'].dropna().unique().astype(int).tolist())}")
        
        mes_zero = (df_work['MES_NUM'] == 0).sum()
        if mes_zero > 0:
            print(f"  Registros con MES=0: {mes_zero:,}")
        
        print("\n Registros por año:")
        year_dist = df_work['ANIO_NUM'].value_counts().sort_index()
        for year, count in year_dist.items():
            if not pd.isna(year):
                print(f"  {int(year)}: {count:>8,} registros")
    
    # 2. Análisis de departamentos
    print("\n--- ANÁLISIS GEOGRÁFICO ---")
    dept_cols = [c for c in df_work.columns if 'DEPTO' in c or 'DEPARTAMENTO' in c]
    if dept_cols:
        dept_col = dept_cols[0]
        depts = df_work[dept_col].dropna().unique()
        print(f"Departamentos únicos: {len(depts)}")
        
        print("\n Top 10 departamentos con más registros:")
        top_depts = df_work[dept_col].value_counts().head(10)
        for dept, count in top_depts.items():
            print(f"  {dept:<20}: {count:>8,}")
    
    # 3. Análisis de montos
    print("\n--- ANÁLISIS DE MONTOS DEVENGADOS ---")
    monto_col = next((c for c in df_work.columns if 'DEVENGADO' in c), None)
    
    if monto_col:
        df_work['MONTO_NUM'] = pd.to_numeric(
            df_work[monto_col].astype(str).str.replace(r'[^0-9\-\.]', '', regex=True),
            errors='coerce'
        )
        
        print(f"Total devengado: S/ {df_work['MONTO_NUM'].sum():,.2f}")
        print(f"Promedio por registro: S/ {df_work['MONTO_NUM'].mean():,.2f}")
        print(f"Mediana: S/ {df_work['MONTO_NUM'].median():,.2f}")
        print(f"Mínimo: S/ {df_work['MONTO_NUM'].min():,.2f}")
        print(f"Máximo: S/ {df_work['MONTO_NUM'].max():,.2f}")
        
        # Montos negativos (reversiones)
        negativos = (df_work['MONTO_NUM'] < 0).sum()
        monto_neg = df_work[df_work['MONTO_NUM'] < 0]['MONTO_NUM'].sum()
        print(f"\nMontos negativos (reversiones): {negativos:,} registros")
        print(f"   Total reversiones: S/ {monto_neg:,.2f}")
        print(f"   Nota: Las reversiones son correcciones presupuestales legítimas")
        
        # Montos extremadamente altos
        Q3 = df_work['MONTO_NUM'].quantile(0.75)
        IQR = Q3 - df_work['MONTO_NUM'].quantile(0.25)
        high_outliers = (df_work['MONTO_NUM'] > (Q3 + 3 * IQR)).sum()
        print(f"Outliers altos (>3*IQR): {high_outliers:,} ({high_outliers/len(df)*100:.2f}%)")


def generate_summary_report(den_stats, ejec_stats):
    """Genera resumen ejecutivo"""
    print(f"\n{'='*60}")
    print("RESUMEN EJECUTIVO - CALIDAD DE DATOS")
    print('='*60)
    
    print("\nRECOMENDACIONES DE LIMPIEZA:")
    print("\nDENUNCIAS POLICIALES:")
    print("  ✓ Eliminar registros con MES=0")
    print("  ✓ Filtrar departamentos con nombres < 3 caracteres")
    print("  ✓ Eliminar cantidades negativas")
    print("  ✓ Tratar outliers extremos (> 5 desviaciones estándar)")
    print("  ✓ Eliminar duplicados exactos")
    print("  ✓ Normalizar nombres de departamentos (mayúsculas, trim)")
    
    print("\nEJECUCIÓN PRESUPUESTAL:")
    print("  ✓ Eliminar registros con MES=0")
    print("  ✓ Filtrar departamentos inválidos")
    print("  ✓ CONSERVAR montos negativos (son reversiones legítimas)")
    print("  ✓ Tratar outliers extremos por departamento")
    print("  ✓ Eliminar duplicados")
    print("  ✓ Normalizar nombres de departamentos")
    
    print("\n🔗 INTEGRACIÓN:")
    print("  ✓ Agregar ambos datasets por (DEPARTAMENTO, MES, AÑO)")
    print("  ✓ Hacer merge en departamentos comunes")
    print("  ✓ Crear panel balanceado con índice temporal completo")
    print("  ✓ Rellenar gaps con forward-fill o 0 según corresponda")
    
    print("\nFEATURES SUGERIDOS:")
    print("  ✓ Lags de ejecución presupuestal (1, 2, 3 meses)")
    print("  ✓ Variables temporales (mes, año)")
    print("  ✓ Target: log1p(cantidad_denuncias) para estabilizar varianza")


def main():
    """Pipeline principal de análisis de calidad"""
    print("="*80)
    print("ANÁLISIS DE CALIDAD DE DATOS - PROYECTO ACADÉMICO")
    print("Regresión: Ejecución Presupuestal PP0030 vs Denuncias Policiales")
    print("="*80)
    
    # 1. Cargar datasets
    print("\nPASO 1: CARGA DE DATOS")
    den = load_raw_data(DENUNCIAS_F, "Denuncias Policiales")
    ejec = load_raw_data(EJEC_F, "Ejecución Presupuestal")
    
    # 2. Análisis de completitud
    print("\nPASO 2: ANÁLISIS DE COMPLETITUD")
    den_missing = analyze_completeness(den, "Denuncias")
    ejec_missing = analyze_completeness(ejec, "Ejecución")
    
    # 3. Análisis de duplicados
    print("\nPASO 3: ANÁLISIS DE DUPLICADOS")
    den_dupes = analyze_duplicates(den, "Denuncias", 
                                   key_columns=['ANIO', 'MES'] if 'ANIO' in den.columns else None)
    ejec_dupes = analyze_duplicates(ejec, "Ejecución",
                                    key_columns=['ANO_EJE', 'MES_EJE'] if 'ANO_EJE' in ejec.columns else None)
    
    # 4. Análisis de tipos de datos
    print("\nPASO 4: ANÁLISIS DE TIPOS DE DATOS")
    den_types = analyze_data_types(den, "Denuncias")
    ejec_types = analyze_data_types(ejec, "Ejecución")
    
    # 5. Análisis específico por dataset
    analyze_denuncias_specific(den)
    analyze_ejecucion_specific(ejec)
    
    # 6. Resumen ejecutivo
    generate_summary_report(
        {'missing': den_missing, 'dupes': den_dupes, 'types': den_types},
        {'missing': ejec_missing, 'dupes': ejec_dupes, 'types': ejec_types}
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS DE CALIDAD COMPLETADO")
    print("="*80)
    print("\nPróximo paso: Ejecutar 'exploratory_regression.py' para limpieza y modelado")


if __name__ == '__main__':
    main()
