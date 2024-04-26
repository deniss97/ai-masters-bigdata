#!/opt/conda/envs/dsenv/bin/python

import os
import sys
import logging
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump

from model import model, fields

logging.basicConfig(level=logging.DEBUG)

try:
    c_param = float(sys.argv[1])
    train_path = sys.argv[2]
except:
    logging.critical("Need to pass both regularization C parameter and train dataset path")
    sys.exit(1)

def configure_model(c):
    model.set_params(linearclassifier__C=c)

configure_model(c_param)

df = pd.read_table(train_path, sep="\t", names=fields, index_col=False)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,2:-1], df.iloc[:,1], test_size=0.33, random_state=42)

with mlflow.start_run():
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)

    loss = log_loss(y_test, predictions)

    mlflow.log_param("C", c_param)
    mlflow.log_metric("log_loss", loss)
    mlflow.sklearn.log_model(model, "model")

    logging.info(f"Model trained with log_loss: {loss:.3f}")

dump(model, "model.joblib")

