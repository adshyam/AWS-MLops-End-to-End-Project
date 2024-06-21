import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    "AdjSquareFeet",
    "DistancetoCoast",
    "DistancetoSinkhole",
    "DistancetoFireDepartment",
    "LocationWindSpeed",
    "Terrain",
    "NumberOfBuildings",
    "NumberOfUnits",
    "Age",
]
label_column = "ValueofHome"

feature_columns_dtype = {
    "AdjSquareFeet": np.float64,
    "DistancetoCoast": np.float64,
    "DistancetoSinkhole": np.float64,
    "DistancetoFireDepartment": np.float64,
    "LocationWindSpeed": np.float64,
    "Terrain": str,
    "NumberOfBuildings": np.float64,
    "NumberOfUnits": np.float64,
    "Age": np.float64,
}
label_column_dtype = {"ValueofHome": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/cleaned_data_no_header.csv",
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    numeric_features = list(feature_columns_names)
    numeric_features.remove("Terrain") #since this is a string
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["Terrain"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    y = df.pop("ValueofHome")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
