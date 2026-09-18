"""
RA1.C - Técnicas de procesamiento de datos.

Análisis exploratorio básico del dataset California Housing:
1. Comprueba si hay valores nulos.
2. Representa gráficamente la distribución de la variable objetivo MedHouseVal.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

AQUI = os.path.dirname(os.path.abspath(__file__))


def cargar_datos():
    """Carga el dataset California Housing como DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame


def comprobar_nulos(df):
    """Comprueba y muestra los valores nulos de cada columna."""
    nulos = df.isnull().sum()
    print("=== COMPROBACIÓN DE VALORES NULOS ===")
    print(nulos)
    print(f"\nTotal de valores nulos: {nulos.sum()}")
    return nulos


def graficar_distribucion(df):
    """Genera y guarda el histograma de MedHouseVal."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["MedHouseVal"], bins=50, edgecolor="black", alpha=0.8)
    ax.set_title("Distribución de la variable objetivo MedHouseVal")
    ax.set_xlabel("MedHouseVal (precio medio, cientos de miles de $)")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    archivo = os.path.join(AQUI, "histograma_medhouseval.png")
    fig.savefig(archivo, dpi=150)
    print(f"\nHistograma guardado en: {archivo}")


def main():
    df = cargar_datos()
    print(f"Dimensiones del dataset: {df.shape}\n")
    comprobar_nulos(df)
    graficar_distribucion(df)


if __name__ == "__main__":
    main()