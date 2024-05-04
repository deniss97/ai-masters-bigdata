import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
from sklearn.externals import joblib

def predict(model_path, test_data, output_path):
    spark = SparkSession.builder.appName("Prediction").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    
    df_test = spark.read.json(test_data)
    
    model = joblib.load(model_path)
    
    features = pd.DataFrame(df_test.select("features").collect())
    predictions = model.predict(features)

    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_path, index_label="id", header=False)

    spark.stop()

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3])

