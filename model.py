import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
URL_AUTOS = "https://raw.githubusercontent.com/mwaskom/seaborn-data/71e2436a092d714350de0fc409ca8a8714e7e78f/mpg.csv"

ENTRADAS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]
NUMERICAS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight_kg",
    "acceleration",
    "model_year",
]

EJEMPLO = {
    "cylinders": 4,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2500.0,
    "acceleration": 16.0,
    "model_year": 78,
    "origin": "usa",
}


def preparar_autos(tabla):
    tabla = tabla[ENTRADAS].copy()
    tabla["horsepower"] = pd.to_numeric(tabla["horsepower"], errors="coerce")
    tabla.loc[tabla["horsepower"] <= 0, "horsepower"] = np.nan
    tabla["origin"] = tabla["origin"].str.strip().str.lower()
    tabla["weight_kg"] = tabla["weight"] * 0.45359237
    return tabla


def construir_pipeline(modelo="knn", vecinos=5):
    numericas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    categoricas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columnas = ColumnTransformer(
        [
            ("numericas", numericas, NUMERICAS),
            ("categoricas", categoricas, ["origin"]),
        ],
        remainder="drop",
    )

    if modelo == "dummy":
        estimador = DummyRegressor(strategy="median")
    else:
        estimador = KNeighborsRegressor(n_neighbors=vecinos)

    return Pipeline(
        [
            ("preparar", FunctionTransformer(preparar_autos, validate=False)),
            ("columnas", columnas),
            ("modelo", estimador),
        ]
    )