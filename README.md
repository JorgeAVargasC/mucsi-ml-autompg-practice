# mucsi-ml-autompg-practice

Regresión de `mpg` (consumo) sobre el dataset Auto MPG con un pipeline de scikit-learn: baseline `DummyRegressor` vs. `KNeighborsRegressor`, evaluados con MAE en validación cruzada (3 folds).

## Estructura

```
model.py            # constantes, limpieza (prepare_cars) y build_pipeline
train.py            # descarga datos, CV, entrenamiento final y predicción de ejemplo
start-mlflow.sh     # levanta el servidor de MLflow
tracking/           # base de datos y artefactos de MLflow (se crea sola)
```

## Comandos

Todos se ejecutan desde la raíz del repo.

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Levantar MLflow

```bash
bash start-mlflow.sh
```

UI en `http://localhost:5000`. En Codespaces, el host del puerto reenviado se detecta solo; para usar otro:

```bash
MLFLOW_HOST=otro.host bash start-mlflow.sh
```

### Entrenar

```bash
# baseline
python train.py --model dummy

# KNN (por defecto 5 vecinos)
python train.py --model knn
python train.py --model knn --neighbors 10
```

Salida:

| Campo | Significado |
|---|---|
| `mae_cv` | MAE medio en validación cruzada |
| `mae_std` | desviación estándar del MAE entre folds |
| `sha256` | hash del CSV descargado (reproducibilidad) |
| `example_prediction` | `mpg` predicho para el coche de `EXAMPLE` |

El KNN debe tener un `mae_cv` claramente menor que el dummy.