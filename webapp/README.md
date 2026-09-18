# California Housing Predictor — Web App

Mini-web **Flask** que sirve el modelo **RandomForestRegressor** de la
práctica RA1.G (California Housing) para que cualquiera introduzca las
**8 características** de una vivienda y obtenga su **precio medio estimado**.

## Estructura

```
webapp/
├── app.py                   # Flask: formulario + API JSON
├── modelo_california.pkl    # modelo entrenado (rosa: recurso RA1.G) ~30 MB
├── model/
│   └── modelo_california.pkl   # copia usada por la app (misma firma)
├── templates/
│   └── index.html           # formulario de 8 campos
├── static/
│   └── style.css            # estilos
├── requirements.txt         # Flask, gunicorn, sklearn, joblib, pandas
├── render.yaml              # config de despliegue en Render (Blueprint)
└── README.md
```

## Correr en local

```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# abre http://127.0.0.1:5000
```

## Predecir

Rellena los 8 campos:
`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`
y pulsa **Calcular**. La web responde con el precio estimado en dólares y
su franja razonable (usando el MAE 0.3277 ≈ 32 770 $ del modelo final).

### API JSON (opcional)

`POST /api/predecir` con JSON:

```json
{
  "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
  "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23
}
```

Devuelve `{precio_usd: 431400, precio_min_usd: ..., precio_max_usd: ...}`.

> Nota: el valor objetivo del dataset está **en cientos de miles de $**
> (MedHouseVal). La web multiplica por 100 000 para mostrar dólares.
