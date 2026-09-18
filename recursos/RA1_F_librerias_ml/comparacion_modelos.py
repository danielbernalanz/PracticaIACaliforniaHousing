"""
RA1.F - Librerías y frameworks de ML
Comparación de dos algoritmos de regresión de scikit-learn:
- LinearRegression (lineal)
- RandomForestRegressor (basado en árboles)

Se comparan en cuanto a velocidad (tiempo de entrenamiento y predicción),
precisión aproximada (R2 y MAE sobre el conjunto de prueba) e interpretabilidad.
"""

import time

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def cargar_datos():
    """Carga el dataset y lo divide en entrenamiento y prueba."""
    housing = fetch_california_housing()
    return train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )


def comparar_modelos():
    """Compara velocidad, precisión e interpretabilidad de dos modelos."""
    X_train, X_test, y_train, y_test = cargar_datos()

    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }

    for nombre, modelo in modelos.items():
        inicio = time.time()
        modelo.fit(X_train, y_train)
        tiempo_entrenamiento = time.time() - inicio

        inicio = time.time()
        y_pred = modelo.predict(X_test)
        tiempo_prediccion = time.time() - inicio

        r2 = modelo.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n=== {nombre} ===")
        print(f"  R2: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Tiempo entrenamiento: {tiempo_entrenamiento:.3f} s")
        print(f"  Tiempo predicción: {tiempo_prediccion:.4f} s")


if __name__ == "__main__":
    comparar_modelos()