from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

def evaluar_modelo_elegido(resultados):
    """
    Selecciona el mejor modelo de CV, lo evalúa en el conjunto de prueba
    y genera los gráficos de validación necesarios para un modelo de regresión.
    """
    mejor_modelo = resultados["cv_results"].iloc[0]["model"]
    print(f"Mejor modelo elegido por Cross-Validation (menor MSE): {mejor_modelo}")
    pipeline_final = resultados["trained_pipelines"][mejor_modelo]

    # X_test y y_test son arrays de NumPy, no tienen información temporal
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]

    y_predicted = pipeline_final.predict(X_test)
    
    # ----------------------------------------------------
    # PASO 1: MÉTRICAS DE DESEMPEÑO
    # ----------------------------------------------------
    print("\n\n📊 MÉTRICAS DE DESEMPEÑO FINAL (Test Set):")
    print(f"Mean squared error (MSE): {mean_squared_error(y_test, y_predicted):.4f}")
    print(f"Mean absolute error (MAE): {mean_absolute_error(y_test, y_predicted):.4f}")
    print(f"R2 score: {r2_score(y_test, y_predicted):.4f}")

    # ----------------------------------------------------
    # PASO 2: GRÁFICO REAL VS PREDICHO (Dispersión)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=[7,7])
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0), alpha=0.6, color='#deff9a') 
    # Línea de identidad (predicción perfecta)
    min_val = min(y_test.min(), y_predicted.min())
    max_val = max(y_test.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Predicción Perfecta")
    ax.set_xlabel('Valor Real (log1p de Denuncias)', fontsize=12)
    ax.set_ylabel('Valor Predicho (log1p de Denuncias)', fontsize=12)
    ax.set_title(f'Real vs. Predicho (Modelo {mejor_modelo})', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.show() 

    print("\n📊 Generando Gráficos de Validación Adicionales...")
    
    # ----------------------------------------------------
    # PASO 3: GRÁFICOS DE RESIDUALES
    # ----------------------------------------------------
    residuos = y_test - y_predicted
    
    # Gráfico 1: Residuales vs Predicciones (Homocedasticidad)
    plot_residue_vs_prediction(y_predicted, residuos, mejor_modelo)
    
    # Gráfico 2: Distribución de Residuos (Normalidad)
    plot_residue_distribution(residuos, mejor_modelo)
    
    # Gráfico 3: Autocorrelación de Residuos (Independencia Temporal)
    # NOTA: Aunque el split es aleatorio, este gráfico es un requisito. Si el split fuera temporal,
    # este gráfico sería la prueba definitiva contra la autocorrelación.
    plot_residue_acf(residuos, mejor_modelo)


def plot_residue_vs_prediction(y_predicted, residuos, model_name):
    """
    Genera el Gráfico de Residuos vs. Predicciones para verificar la Homocedasticidad.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_predicted, residuos, alpha=0.6, edgecolors='w', linewidth=0.5, color='#deff9a') 
    
    # Línea central en cero
    plt.axhline(0, color='red', linestyle='--', linewidth=2) 
    
    plt.title(f"Residuos vs. Predicciones (Homocedasticidad) - Modelo {model_name}", fontsize=14)
    plt.xlabel("Valor Predicho (log1p)", fontsize=12)
    plt.ylabel("Residuo (Error)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Gráfico de Normalidad | Histograma de Residuos o Distribución de Residuos
def plot_residue_distribution(residuos, model_name):
    """
    Genera el Histograma y el QQ-Plot de los residuos (errores)
    para verificar su distribución (normalidad).
    """
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='#0077B6') # Azul
    plt.title(f"Histograma de Residuos - Modelo {model_name}", fontsize=14)
    plt.xlabel("Valor del Residuo", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.axvline(residuos.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f"Media: {residuos.mean():.4f}")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title("QQ-Plot de Residuos (Normalidad)", fontsize=14)
    plt.xlabel("Cuantiles Teóricos", fontsize=11)
    plt.ylabel("Cuantiles de la Muestra", fontsize=11)
    
    plt.tight_layout()
    plt.show()


# Grafico de ACF de Residuos | Autocorrrelacion de residuos (Independencia temporal)  
def plot_residue_acf(residuos, model_name):
    """
    Genera el Gráfico de Función de Autocorrelación (ACF) de los residuos
    para verificar la independencia de los errores a lo largo del tiempo.
    """
    plt.figure(figsize=(10, 5))

    plot_acf(residuos, 
             lags=20, # Mostrar correlación hasta 20 meses
             ax=plt.gca(), 
             title=f"Autocorrelación de Residuos (ACF) - Modelo {model_name}",
             color='#deff9a')
    
    plt.xlabel("Lag (Meses)", fontsize=11)
    plt.ylabel("Coeficiente de Autocorrelación", fontsize=11)
    plt.tight_layout()
    plt.show()