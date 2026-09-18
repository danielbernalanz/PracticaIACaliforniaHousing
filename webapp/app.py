"""
Webapp de la práctica RA1 · California Housing.

Expone el modelo entrenado (RandomForestRegressor, MAE 0.3277) como un
servicio web: el usuario introduce las 8 características de una vivienda
(2 de ellas, Latitud y Longitud, se pueden rellenar clicando en el mapa)
y la web devuelve el precio medio estimado con su franja de error (MAE).
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(AQUI, "model", "modelo_california.pkl")
MAE = 0.3277  # cientos de miles de $ (ver recursos/RA1_G)

# Las 8 características, en EL MISMO ORDEN que X del modelo entrenado.
COLUMNAS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Etiquetas legibles que la plantilla muestra junto a cada campo.
NOMBRES = {
    "MedInc": "Ingreso medio (miles de $)",
    "HouseAge": "Antigüedad (años)",
    "AveRooms": "Habitaciones promedio",
    "AveBedrms": "Dormitorios promedio",
    "Population": "Población del bloque",
    "AveOccup": "Ocupantes promedio",
    "Latitude": "Latitud",
    "Longitude": "Longitud",
}

modelo = None


def obtener_modelo():
    """Carga el modelo una sola vez y lo reutiliza (singleton)."""
    global modelo
    if modelo is None:
        modelo = joblib.load(RUTA_MODELO)
    return modelo


app = Flask(__name__)


@app.route("/")
def inicio():
    """Página principal con el formulario."""
    return render_template(
        "index.html",
        columnas=COLUMNAS,
        nombres=NOMBRES,
        mae_cien_mil=MAE,
        valores=None,
        resultado=None,
        error=None,
    )


@app.route("/predecir", methods=["POST"])
def predecir():
    """Lee los 8 valores del formulario y predice el precio."""
    try:
        fila = {c: float(request.form[c]) for c in COLUMNAS}
        X_nuevo = pd.DataFrame([fila], columns=COLUMNAS)
        precio_cien_mil = float(obtener_modelo().predict(X_nuevo)[0])
        resultado = {
            "precio_cien_mil": round(precio_cien_mil, 4),
            "precio_usd": round(precio_cien_mil * 100000, 0),
            "min_usd": round((precio_cien_mil - MAE) * 100000, 0),
            "max_usd": round((precio_cien_mil + MAE) * 100000, 0),
            "mae_cien_mil": MAE,
            "inputs": fila,
        }
        return render_template(
            "index.html",
            columnas=COLUMNAS,
            nombres=NOMBRES,
            mae_cien_mil=MAE,
            valores=fila,
            resultado=resultado,
            error=None,
        )
    except (ValueError, KeyError) as e:
        return render_template(
            "index.html",
            columnas=COLUMNAS,
            nombres=NOMBRES,
            mae_cien_mil=MAE,
            valores=None,
            resultado=None,
            error=f"Valores no válidos: introduce números correctos ({e}).",
        )


@app.route("/api/predecir", methods=["POST"])
def api_predecir():
    """Mismo servicio en JSON, para POSTman/curl/otros programas."""
    try:
        datos = request.get_json(force=True)
        fila = {c: float(datos[c]) for c in COLUMNAS}
        X_nuevo = pd.DataFrame([fila], columns=COLUMNAS)
        precio_cien_mil = float(obtener_modelo().predict(X_nuevo)[0])
        return jsonify(
            {
                "precio_cien_mil": round(precio_cien_mil, 4),
                "precio_usd": round(precio_cien_mil * 100000, 0),
                "modelo": "RandomForestRegressor",
                "mae_cien_mil": MAE,
            }
        )
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/info")
def api_info():
    """Metadatos del modelo para documentar la API."""
    return jsonify(
        {
            "dataset": "California Housing (20.640 muestras x 9 columnas)",
            "modelo": "RandomForestRegressor",
            "n_estimadores": 100,
            "random_state": 42,
            "columnas": COLUMNAS,
            "mae": MAE,
            "unidades": "cientos de miles de $. precio_usd = precio_cien_mil * 100000",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
