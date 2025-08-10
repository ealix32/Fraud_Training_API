# Fraud Scoring API

Servicio REST en FastAPI para entrenar y usar un modelo de detección de fraude.

## Requisitos

- Python 3.10+
- Instalar dependencias:
  ```bash
  pip install fastapi uvicorn scikit-learn pandas numpy joblib
  ```

## Cómo ejecutar

```bash
uvicorn service:app --reload --port 8000
```

La documentación interactiva está disponible en `/docs`.

## Ejemplos `curl`

Listar features disponibles:

```bash
curl -s http://localhost:8000/features
```

Entrenar con columnas seleccionadas:

```bash
curl -s -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"selected_columns":["Age","VehiclePrice","PolicyType","DayOfWeek","Make"]}'
```

Predecir probabilidad de fraude para un cliente:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "items":[
      {"features":{
        "Age":45,
        "VehiclePrice":28000,
        "PolicyType":"Sedan - Liability",
        "DayOfWeek":"Monday",
        "Make":"Toyota"
      }}
    ]
  }'
```

La respuesta incluye `fraud_probability` como salida principal y `predicted_label`, que usa el umbral óptimo calculado durante `/train`.

## Umbral manual (opcional)

Puedes fijar manualmente el umbral de decisión:

```bash
curl -s -X POST http://localhost:8000/config/threshold \
  -H 'Content-Type: application/json' \
  -d '{"threshold":0.5}'
```
