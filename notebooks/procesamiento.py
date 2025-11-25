# procesamiento.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import time
from tabulate import tabulate


def dividir_datos(df: pd.DataFrame, meses_test: int = 6):
    """
    Divide el panel balanceado en conjuntos de entrenamiento y prueba
    usando un split temporal (últimos N meses para prueba).
    """
    last_period = df["period"].max()
    test_start = (last_period - pd.DateOffset(months=meses_test)) + pd.DateOffset(days=1)

    train = df[df["period"] < test_start].copy()
    test = df[df["period"] >= test_start].copy()

    print(f"[INFO] Último periodo: {last_period}")
    print(f"[INFO] Conjunto de prueba inicia en: {test_start.date()}")
    print(f"[INFO] Tamaño entrenamiento: {len(train)} | Tamaño prueba: {len(test)}")

    return train, test


def evaluar_modelos_cv(
    df_features: pd.DataFrame,
    features: list = None,
    target: str = "y",
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 10,
):
    """
    Realiza un split 80/20 y evalúa múltiples modelos
    usando K-fold CV (K=n_splits) únicamente sobre el conjunto de entrenamiento.

    Retorna métricas promedio por modelo y los pipelines entrenados.
    """
    df = df_features.copy()

    if features is None:
        features = [c for c in df.columns if c.startswith("MONTO_LAG_")] + ["month", "year"]

    # Usamos el split aleatorio aquí para la validación cruzada general
    X = df[features].fillna(0.0).values
    y = df[target].values
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Definir modelos a evaluar. Usamos Ridge, Lasso y ElasticNet con alphas fijos 
    # para la comparación rápida, salvo el RidgeCV inicial del baseline.
    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
    }

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
    
    # Imprimir coeficientes del mejor modelo Lineal (para la discusión de influencia)
    mejor_lineal = cv_df[cv_df['model'].isin(['Ridge', 'Lasso', 'ElasticNet'])].iloc[0]["model"]
    pipeline_lineal = trained_pipelines[mejor_lineal]
    
    print(f"\n\nCOEFICIENTES DEL MEJOR MODELO LINEAL ({mejor_lineal}):")
    # Intentar obtener el estimador del pipeline
    estimator = pipeline_lineal.named_steps.get('est')

    if hasattr(estimator, 'coef_'):
        coefs = pd.DataFrame({
            "Feature": features,
            "Coeficiente": estimator.coef_
        }).sort_values("Coeficiente", key=abs, ascending=False)
        
        coef_table = []
        for _, row in coefs.iterrows():
            direction = "↑ Positivo" if row["Coeficiente"] > 0 else "↓ Negativo"
            coef_table.append([row["Feature"], f"{row['Coeficiente']:.6f}", direction])

        print(tabulate(coef_table,
                        headers=["Feature", "Valor", "Dirección"],
                        tablefmt="fancy_grid"))
    else:
        print("[INFO] El modelo lineal no tiene coeficientes (Error).")


    return {
        "cv_results": cv_df,
        "trained_pipelines": trained_pipelines,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }