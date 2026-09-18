"""
RA1.E - Estructuras de control y funciones.

Organiza el trabajo en dos funciones independientes con una única
responsabilidad cada una:
1. cargar_datos(): carga el dataset California Housing.
2. dividir_datos(): divide los datos en entrenamiento y prueba.

Incluye una pequeña prueba (test) que demuestra que ambas funcionan.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def cargar_datos(as_frame=True):
    """Carga el dataset California Housing y lo devuelve como (X, y)."""
    housing = fetch_california_housing(as_frame=as_frame)
    return housing.data, housing.target


def dividir_datos(X, y, test_size=0.2):
    """Divide X e y en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def test():
    """Prueba sencilla que verifica que las funciones funcionan."""
    X, y = cargar_datos()
    assert X.shape[0] == 20640, "Número de filas inesperado"
    assert X.shape[1] == 8, "Número de características inesperado"
    assert len(y) == 20640, "Las etiquetas no coinciden con los datos"

    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Entrenamiento: X={X_train.shape}, y={y_train.shape}")
    print(f"Prueba:        X={X_test.shape}, y={y_test.shape}")

    assert X_train.shape[0] == 16512, "Tamaño de entrenamiento inesperado"
    assert X_test.shape[0] == 4128, "Tamaño de prueba inesperado"
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]

    print("\nPrueba superada: las funciones cargan y dividen correctamente.")


if __name__ == "__main__":
    test()