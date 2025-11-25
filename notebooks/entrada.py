"""
===================================================================================
MÓDULO: ENTRADA DE DATOS (Data Input)
===================================================================================

Responsabilidades:
- Lectura de archivos CSV con múltiples encodings
- Normalización inicial de nombres de columnas
- Agregación temporal por (AÑO, MES, DEPARTAMENTO)
- Validación de estructura básica

Output: DataFrames limpios a nivel de estructura con columnas estandarizadas
===================================================================================
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def read_denuncias(filepath: str) -> pd.DataFrame:
    """
    Lee el dataset de denuncias policiales con detección automática de encoding.
    
    Args:
        filepath: Ruta al archivo CSV de denuncias
        
    Returns:
        DataFrame con columnas [DEPARTAMENTO, period, CANTIDAD]
        
    Proceso:
    1. Detecta encoding automáticamente
    2. Normaliza nombres de columnas
    3. Filtra MES != 0 (totales anuales)
    4. Agrega por (ANIO, MES, DEPARTAMENTO)
    """
    print("\n" + "="*80)
    print(" "*30 + "ENTRADA: DENUNCIAS POLICIALES")
    print("="*80)
    
    # 1. Lectura con múltiples encodings
    encodings = ["utf-8-sig", "latin1", "cp1252", "utf-8", "iso-8859-1"]
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"Archivo leído exitosamente con encoding: {enc}")
            break
        except Exception as e:
            continue
    
    if df is None:
        raise ValueError(f"No se pudo leer el archivo con ningún encoding probado: {encodings}")
    
    print(f"   • Filas leídas: {len(df):,}")
    print(f"   • Columnas: {list(df.columns)}")
    
    # 2. Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.upper()
    
    # Detectar columnas relevantes (pueden tener variaciones)
    year_col = next((c for c in df.columns if "ANO" in c or "ANIO" in c), None)
    month_col = next((c for c in df.columns if "MES" in c), None)
    dept_col = next((c for c in df.columns if "DPTO" in c or "DEPART" in c), None)
    qty_col = next((c for c in df.columns if "CANTIDAD" in c or "TOTAL" in c), None)
    
    if not all([year_col, month_col, dept_col, qty_col]):
        raise ValueError(f"No se encontraron todas las columnas necesarias en denuncias")
    
    print(f"   • Columnas detectadas: AÑO={year_col}, MES={month_col}, DPTO={dept_col}, CANT={qty_col}")
    
    # 3. Renombrar a nombres estándar
    df = df[[year_col, month_col, dept_col, qty_col]].copy()
    df.columns = ["ANIO", "MES", "DEPARTAMENTO", "CANTIDAD"]
    
    # 4. Filtrar MES = 0 (son totales anuales, no datos mensuales)
    before = len(df)
    df = df[df["MES"] != 0].copy()
    print(f"   • Filas con MES=0 eliminadas: {before - len(df):,} (totales anuales)")
    
    # 5. Convertir tipos
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce")
    
    # 6. Crear columna de periodo
    df["period"] = pd.to_datetime(
        df["ANIO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    
    # 7. Agregar por (ANIO, MES, DEPARTAMENTO) - elimina duplicados por diseño
    df = df.groupby(["DEPARTAMENTO", "period"], as_index=False)["CANTIDAD"].sum()
    
    print(f"   • Filas después de agregación: {len(df):,}")
    print(f"   • Departamentos únicos: {df['DEPARTAMENTO'].nunique()}")
    print(f"   • Rango temporal: {df['period'].min().date()} a {df['period'].max().date()}")
    
    return df


def read_ejecucion(filepath: str) -> pd.DataFrame:
    """
    Lee el dataset de ejecución presupuestal PP0030 con detección automática de encoding.
    
    Args:
        filepath: Ruta al archivo CSV de ejecución presupuestal
        
    Returns:
        DataFrame con columnas [DEPARTAMENTO, period, MONTO_DEVENGADO]
        
    Proceso:
    1. Detecta encoding automáticamente
    2. Normaliza nombres de columnas
    3. Detecta variantes de columnas (ANO_EJE/AÑO, etc.)
    4. Limpia montos (quita símbolos de moneda)
    5. Agrega por (ANIO, MES, DEPARTAMENTO)
    """
    print("\n" + "="*80)
    print(" "*25 + "ENTRADA: EJECUCIÓN PRESUPUESTAL PP0030")
    print("="*80)
    
    # 1. Lectura con múltiples encodings
    encodings = ["utf-8-sig", "latin1", "cp1252", "utf-8", "iso-8859-1"]
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"Archivo leído exitosamente con encoding: {enc}")
            break
        except Exception as e:
            continue
    
    if df is None:
        raise ValueError(f"No se pudo leer el archivo con ningún encoding probado: {encodings}")
    
    print(f"   • Filas leídas: {len(df):,}")
    print(f"   • Columnas: {list(df.columns)}")
    
    # 2. Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.upper()
    
    # 3. Detectar columnas relevantes (pueden tener variaciones)
    year_col = next((c for c in df.columns if "ANO" in c and "EJE" in c), None)
    month_col = next((c for c in df.columns if "MES" in c and "EJE" in c), None)
    dept_col = next((c for c in df.columns if "DEPTO" in c or "DEPART" in c), None)
    # IMPORTANTE: Priorizar MONTO_DEVENGADO sobre otras columnas con "MONTO"
    monto_col = next((c for c in df.columns if "DEVENGADO" in c), None)
    if monto_col is None:
        monto_col = next((c for c in df.columns if "MONTO" in c), None)
    
    if not all([year_col, month_col, dept_col, monto_col]):
        raise ValueError(f"No se encontraron todas las columnas necesarias en ejecución presupuestal")
    
    print(f"   • Columnas detectadas: AÑO={year_col}, MES={month_col}, DPTO={dept_col}, MONTO={monto_col}")
    
    # 4. Renombrar a nombres estándar
    df = df[[year_col, month_col, dept_col, monto_col]].copy()
    df.columns = ["ANIO", "MES", "DEPARTAMENTO", "MONTO_DEVENGADO"]
    
    # 5. Limpiar columna de montos (puede tener símbolos S/, comas, etc.)
    if df["MONTO_DEVENGADO"].dtype == "object":
        df["MONTO_DEVENGADO"] = (
            df["MONTO_DEVENGADO"]
            .astype(str)
            .str.replace("S/", "", regex=False)
            .str.replace(",", "")
            .str.strip()
        )
    
    # 6. Convertir tipos
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
    df["MONTO_DEVENGADO"] = pd.to_numeric(df["MONTO_DEVENGADO"], errors="coerce")
    
    # 7. Crear columna de periodo
    df["period"] = pd.to_datetime(
        df["ANIO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    
    # 8. Agregar por (ANIO, MES, DEPARTAMENTO)
    df = df.groupby(["DEPARTAMENTO", "period"], as_index=False)["MONTO_DEVENGADO"].sum()
    
    print(f"   • Filas después de agregación: {len(df):,}")
    print(f"   • Departamentos únicos: {df['DEPARTAMENTO'].nunique()}")
    print(f"   • Rango temporal: {df['period'].min().date()} a {df['period'].max().date()}")
    print(f"   • Monto total: S/ {df['MONTO_DEVENGADO'].sum():,.2f}")
    
    return df


def load_data(data_dir: str = "../data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga ambos datasets desde el directorio especificado.
    
    Args:
        data_dir: Directorio donde se encuentran los CSVs
        
    Returns:
        Tupla (df_denuncias, df_ejecucion)
    """
    data_path = Path(data_dir)
    
    # Buscar archivos
    denuncias_file = next(data_path.glob("*Denuncias*.csv"), None)
    ejecucion_file = next(data_path.glob("*Ejecu_Presup*.csv"), None)
    
    if not denuncias_file:
        raise FileNotFoundError(f"No se encontró archivo de denuncias en {data_dir}")
    if not ejecucion_file:
        raise FileNotFoundError(f"No se encontró archivo de ejecución presupuestal en {data_dir}")
    
    print("="*80)
    print(" "*25 + "INICIANDO CARGA DE DATOS")
    print("="*80)
    print(f"Directorio: {data_path.absolute()}")
    print(f"Denuncias: {denuncias_file.name}")
    print(f"Ejecución: {ejecucion_file.name}")
    
    df_denuncias = read_denuncias(str(denuncias_file))
    df_ejecucion = read_ejecucion(str(ejecucion_file))
    
    print("\n" + "="*80)
    print(" "*30 + "CARGA COMPLETADA")
    print("="*80)
    
    return df_denuncias, df_ejecucion


if __name__ == "__main__":
    # Prueba del módulo
    df_den, df_eje = load_data("../data/raw")
    print("\nDenuncias shape:", df_den.shape)
    print("Ejecución shape:", df_eje.shape)
