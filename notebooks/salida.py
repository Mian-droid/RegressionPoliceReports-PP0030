from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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
