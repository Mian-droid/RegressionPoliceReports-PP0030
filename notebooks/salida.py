from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np  
from statsmodels.graphics.tsaplots import plot_acf

def evaluar_modelo_elegido(resultados):
    mejor_modelo = resultados["cv_results"].iloc[0]["model"]
    print(f"Mejor modelo: {mejor_modelo}")
    pipeline_final = resultados["trained_pipelines"][mejor_modelo]

    X_test = resultados["X_test"]
    y_test = resultados["y_test"]

    y_predicted = pipeline_final.predict(X_test)
    print("Mean squared error (MSE):", mean_squared_error(y_test, y_predicted))
    print("Mean absolute error (MAE):", mean_absolute_error(y_test, y_predicted))
    print("R2 score:", r2_score(y_test, y_predicted))

    fig, ax = plt.subplots( figsize=[7,7])
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicho')
    plt.show() 

    print("\n📊 Generando Gráficos de Validación Adicionales...")
    
    # Gráfico 1: Residuales vs Predicciones
    plot_residue_vs_prediction(resultados, pipeline_final)
    
    # Gráfico 2: Distribución de Residuos (Normalidad)
    plot_residue_distribution(resultados, pipeline_final)
    
    # Gráfico 3: Autocorrelación de Residuos (Independencia Temporal)
    #plot_residue_acf(resultados, pipeline_final)


def plot_residue_vs_prediction(resultados, pipeline_final):
    """
    Genera el Gráfico de Residuos vs. Predicciones para verificar la Homocedasticidad.
    """
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]

    # 1. Obtener predicciones y residuos
    y_predicted = pipeline_final.predict(X_test)
    residuos = y_test - y_predicted
    
    # 2. Generar el gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(y_predicted, residuos, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Línea central en cero
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Residuos vs. Predicciones (Homocedasticidad) del Modelo {resultados['cv_results'].iloc[0]['model']}", fontsize=14)
    plt.xlabel("Valor Predicho (log1p)", fontsize=12)
    plt.ylabel("Residuo (Error)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

#Gráfico de Normalidad | Histograma de Residuos o Distribución de Residuos
def plot_residue_distribution(resultados, pipeline_final):
    """
    Genera el Histograma y el QQ-Plot de los residuos (errores)
    para verificar su distribución (normalidad).
    """
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]

    residuos = y_test - pipeline_final.predict(X_test)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Histograma de Residuos del Modelo {resultados['cv_results'].iloc[0]['model']}", fontsize=14)
    plt.xlabel("Valor del Residuo", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.axvline(residuos.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f"Media: {residuos.mean():.4f}")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title("QQ-Plot de Residuos", fontsize=14)
    plt.xlabel("Cuantiles Teóricos", fontsize=11)
    plt.ylabel("Cuantiles de la Muestra", fontsize=11)
    
    plt.tight_layout()
    plt.show()


#Grafico de ACF de Residuos | Autocorrrelacion de residuos (Independencia temporal)  
def plot_residue_acf(resultados, pipeline_final):
    """
    Genera el Gráfico de Función de Autocorrelación (ACF) de los residuos
    para verificar la independencia de los errores a lo largo del tiempo.
    """
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]
    residuos = y_test - pipeline_final.predict(X_test)
    plt.figure(figsize=(10, 5))

    plot_acf(residuos, 
             lags=20, # Mostrar correlación hasta 20 meses
             ax=plt.gca(), 
             title=f"Autocorrelación de Residuos del Modelo {resultados['cv_results'].iloc[0]['model']}")
    
    plt.xlabel("Lag (Meses)", fontsize=11)
    plt.ylabel("Coeficiente de Autocorrelación", fontsize=11)
    plt.tight_layout()
    plt.show()