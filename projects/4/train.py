import sys
from pyspark.sql import SparkSession

def train(input_data, model_path):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    from model import pipeline
    df = spark.read.json(input_data)
    df = spark.read.json(input_data).na.fill('No Review', subset=["reviewText"])
    model = pipeline.fit(df)
    model.write().overwrite().save(model_path)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])

