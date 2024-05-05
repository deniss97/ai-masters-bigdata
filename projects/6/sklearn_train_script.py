import sys
import argparse
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model(train_in, sklearn_model_out):

    file_paths = glob.glob(train_dir + "/*.json")
    data_frames = []

    for file_path in file_paths:
        df = pd.read_json(file_path, lines=True)
        data_frames.append(df)

    data = pd.concat(data_frames, ignore_index=True)
    # data = pd.read_json(train_in, lines=True)
    features = data["reviewText"].tolist()
    labels = data['label'].tolist()

    model = LinearRegression()
    model.fit(features, labels)

    joblib.dump(model, sklearn_model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-in', type=str, required=True)
    parser.add_argument('--sklearn-model-out', type=str, required=True)
    args = parser.parse_args()

    train_model(args.train_in, args.sklearn_model_out)

