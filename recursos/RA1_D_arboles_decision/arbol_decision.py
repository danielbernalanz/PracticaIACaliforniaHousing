"""
RA1.D - Estructuras de datos dinámicas, árboles y/o grafos.

Entrena un árbol de decisión pequeño (max_depth=3) sobre el dataset California
Housing y lo visualiza para entender su estructura interna: nodos de decisión
y hojas.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

AQUI = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


def main():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    arbol = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    arbol.fit(X_train, y_train)

    print(f"Número de nodos: {arbol.tree_.node_count}")
    print(f"Número de hojas: {arbol.tree_.n_leaves}")
    print(f"Profundidad máxima: {arbol.tree_.max_depth}")

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        arbol,
        feature_names=X.columns.tolist(),
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    fig.tight_layout()
    archivo = os.path.join(AQUI, "arbol_decision.png")
    fig.savefig(archivo, dpi=150)
    print(f"Árbol guardado en: {archivo}")


if __name__ == "__main__":
    main()