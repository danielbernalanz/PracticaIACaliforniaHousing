"""
RA1.G - Desarrollo y entrenamiento de un modelo simple.

Entrena el modelo elegido (RandomForestRegressor) con los datos de
entrenamiento, evalúa su error sobre los datos de prueba con el error
absoluto medio (MAE) y guarda el modelo entrenado en disco con joblib.
"""

import os

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

AQUI = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


def main():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE sobre el conjunto de prueba: {mae:.4f}")

    ruta_modelo = os.path.join(AQUI, "modelo_california.pkl")
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    main()