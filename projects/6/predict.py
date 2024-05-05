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
    
    features = pd.DataFrame(df_test.select("reviewText").collect())
    predictions = model.predict(features)

    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_path, index_label="id", header=False)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1
    
    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

