import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer

def feature_engineering(input_path, output_path):
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    
    df = spark.read.json(input_path).na.fill('No Review', subset=["reviewText"])

    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="features")
    indexer = StringIndexer(inputCol="overall", outputCol="label")  # Используем overall как метку

    pipeline = Pipeline(stages=[tokenizer, hashingTF, indexer])
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    
    transformed_df.write.mode("overwrite").json(output_path)

    spark.stop()

if __name__ == "__main__":
    feature_engineering(sys.argv[1], sys.argv[2])

