import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

def predict(model_path, test_data, output_path):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df_test = spark.read.json(test_data)
    model = PipelineModel.load(model_path)
    predictions = model.transform(df_test)
    predictions.write.json(output_path)

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3])

