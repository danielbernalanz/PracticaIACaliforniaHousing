# California Housing Predictor — Web (Flask)

Mini-web que expone el modelo **RandomForestRegressor** entrenado en la
práctica RA1 (California Housing) para que cualquiera introduzca las
**8 características** de una vivienda y obtenga su **precio medio estimado**.

## Cómo funciona

- Carga `model/modelo_california.pkl` (modelo exacto de RA1.G, comprimido
  con joblib `compress=("gzip",9)` para caber en GitHub: ~30 MB).
- El usuario rellena un formulario con las 8 columnas del dataset
  (`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
  Latitude, Longitude`) en **el mismo orden** en que el modelo fue entrenado.
- `RandomForestRegressor.predict()` devuelve el precio en **cientos de
  miles de $**; la web lo convierte a dólares y muestra el rango usando el
  **MAE 0.3277** de la práctica.

## Ejecutar en local

```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# abre http://127.0.0.1:5000
```

## Desplegar en Render

1. Sube el proyecto a GitHub (este repo ya está conectado).
2. En [render.com](https://render.com) → **New → Blueprint** → selecciona
   este repo. Render lee `webapp/render.yaml` automáticamente.
3. Espera el despliegue (~2–3 min). La web queda en
   `https://california-housing-predictor.onrender.com`.
4. La primera carga tarda unos segundos (el servicio free "duerme" tras
   15 min de inactividad y se despierta bajo demanda).

## Rutas

- `GET /` → formulario con las 8 características.
- `POST /predecir` → envío del formulario, devuelve el precio estimado.
- `GET /api/predecir` → misma predicción en JSON (para clientes API).

## Requisitos

Ver `requirements.txt` (Flask, joblib, numpy, pandas, scikit-learn).
