import argparse
import hashlib
import io
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from model import EJEMPLO, ENTRADAS, SEED, URL_AUTOS, construir_pipeline


def train(name, neighbors):
    with urlopen(URL_AUTOS, timeout=30) as res:
        content = res.read()
    digest = hashlib.sha256(content).hexdigest()
    data = pd.read_csv(io.BytesIO(content))

    dev, _ = train_test_split(data, test_size=0.2, random_state=SEED)
    X, y = dev[ENTRADAS], dev["mpg"]

    folds = list(KFold(n_splits=3, shuffle=True, random_state=SEED).split(X))

    pipeline = construir_pipeline(name, neighbors)

    maes = -cross_val_score(
        pipeline,
        X,
        y,
        cv=folds,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        error_score="raise",
    )

    pipeline.fit(X, y)

    example = pd.DataFrame([EJEMPLO])
    for col in example.columns.difference(["origin"]):
        example[col] = example[col].astype(float)

    return pipeline, maes, digest, example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dummy", "knn"], default="knn")
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    pipeline, maes, digest, example = train(args.model, args.neighbors)

    print(
        {
            "model": args.model,
            "neighbors": args.neighbors if args.model == "knn" else None,
            "mae_cv": float(maes.mean()),
            "desviacion": float(maes.std()),
            "sha256": digest,
            "prediccion_ejemplo": float(pipeline.predict(example)[0]),
        }
    )
