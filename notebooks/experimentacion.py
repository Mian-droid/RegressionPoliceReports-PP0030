#experimentacion.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def grafica_box_plot(resultados):
    df = resultados

    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle("comparación de R^2 promedio (10-Folds CV)")

    ax.bar(df["model"], df["r2_mean"], yerr=df["r2_std"], capsize=5, color="skyblue", edgecolor="black")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("R^2 promedio")
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    
