# procesamiento.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
import time


def dividir_datos(df: pd.DataFrame, meses_test: int = 6):
    """
    Divide el panel balanceado en conjuntos de entrenamiento y prueba
    usando un split temporal (últimos N meses para prueba).

    Args:
        df (pd.DataFrame): DataFrame con las columnas ['period', 'y', features...]
        meses_test (int): Número de meses recientes usados para el conjunto de prueba.

    Returns:
        tuple: (train_df, test_df)
    """
    last_period = df["period"].max()
    test_start = (last_period - pd.DateOffset(months=meses_test)) + pd.DateOffset(days=1)

    train = df[df["period"] < test_start].copy()
    test = df[df["period"] >= test_start].copy()

    print(f"[INFO] Último periodo: {last_period}")
    print(f"[INFO] Conjunto de prueba inicia en: {test_start.date()}")
    print(f"[INFO] Tamaño entrenamiento: {len(train)} | Tamaño prueba: {len(test)}")

    return train, test


def entrenar_modelo(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, target: str = "y"):
    """
    Entrena un modelo Ridge Regression con validación cruzada (K=10),
    usando estandarización de variables y evalúa el rendimiento.

    Args:
        train_df (pd.DataFrame): Datos de entrenamiento
        test_df (pd.DataFrame): Datos de prueba
        features (list): Lista de nombres de columnas predictoras
        target (str): Nombre de la variable objetivo

    Returns:
        dict: Resultados del modelo, incluyendo métricas y predicciones
    """
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Definir pipeline: estandarización + regresión ridge con validación cruzada
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=10))
    ])

    # Entrenar modelo
    pipeline.fit(X_train, y_train)

    # Predicciones
    preds_train = pipeline.predict(X_train)
    preds_test = pipeline.predict(X_test)

    # Evaluación
    metrics = {
        "train_rmse": mean_squared_error(y_train, preds_train, squared=False),
        "train_mae": mean_absolute_error(y_train, preds_train),
        "train_r2": r2_score(y_train, preds_train),
        "test_rmse": mean_squared_error(y_test, preds_test, squared=False),
        "test_mae": mean_absolute_error(y_test, preds_test),
        "test_r2": r2_score(y_test, preds_test)
    }

    print("\n[INFO] Métricas de rendimiento:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Consolidar resultados
    resultados = {
        "modelo": pipeline,
        "metricas": metrics,
        "predicciones": {
            "train": pd.DataFrame({"real": y_train, "pred": preds_train}, index=X_train.index),
            "test": pd.DataFrame({"real": y_test, "pred": preds_test}, index=X_test.index),
        }
    }

    return resultados


def ejecutar_procesamiento(df_features: pd.DataFrame):
    """
    Función principal para ejecutar todo el flujo de procesamiento:
    1. División temporal
    2. Entrenamiento y validación
    3. Devolución de métricas y modelo final

    Args:
        df_features (pd.DataFrame): Datos con las columnas ['period', 'y', features...]

    Returns:
        dict: Resultados del modelo (métricas, predicciones, objeto pipeline)
    """
    # Variables predictoras
    features = ["MONTO_LAG_1", "MONTO_LAG_2", "MONTO_LAG_3", "month", "year"]

    train_df, test_df = dividir_datos(df_features)
    resultados = entrenar_modelo(train_df, test_df, features)

    return resultados


def evaluar_modelos_cv(
    df_features: pd.DataFrame,
    features: list = None,
    target: str = "y",
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 10,
):
    """
    Realiza un split 80/20 (por defecto) y evalúa múltiples modelos
    usando K-fold CV (K=n_splits) únicamente sobre el conjunto de entrenamiento.

    Retorna métricas promedio (MSE, MAE, R2) por modelo y los pipelines
    entrenados sobre el 80% (training set completo) para uso posterior.
    """
    df = df_features.copy()

    if features is None:
        features = [c for c in df.columns if c.startswith("MONTO_LAG_")] + ["month", "year"]

    X = df[features].fillna(0.0).values
    y = df[target].values

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Definir modelos a evaluar
    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
    }

    # K-Fold CV sobre el conjunto de entrenamiento
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = []
    trained_pipelines = {}

    for name, est in modelos.items():
        print(f"[INFO] Evaluando modelo: {name}")
        fold_mses = []
        fold_maes = []
        fold_r2s = []
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            # Crear pipeline para cada fold (Scaler + estimator)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("est", est)
            ])

            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)

            fold_mses.append(mean_squared_error(y_val, preds))
            fold_maes.append(mean_absolute_error(y_val, preds))
            fold_r2s.append(r2_score(y_val, preds))

        elapsed = time.time() - start_time

        # Promedio y desviación
        result = {
            "model": name,
            "mse_mean": float(np.mean(fold_mses)),
            "mse_std": float(np.std(fold_mses)),
            "mae_mean": float(np.mean(fold_maes)),
            "mae_std": float(np.std(fold_maes)),
            "r2_mean": float(np.mean(fold_r2s)),
            "r2_std": float(np.std(fold_r2s)),
            "cv_time_sec": elapsed,
            "n_folds": n_splits,
        }

        cv_results.append(result)

        # Entrenar pipeline final sobre todo el conjunto de entrenamiento (80%)
        final_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("est", est)
        ])
        final_pipeline.fit(X_train, y_train)
        trained_pipelines[name] = final_pipeline

    cv_df = pd.DataFrame(cv_results).sort_values("mse_mean")

    return {
        "cv_results": cv_df,
        "trained_pipelines": trained_pipelines,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }