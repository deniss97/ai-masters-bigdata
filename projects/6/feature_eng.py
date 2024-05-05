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
    # indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")

    pipeline = Pipeline(stages=[tokenizer, hashingTF])
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    
    transformed_df.write.mode("overwrite").json(output_path)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--path-in") + 1
    output_arg_index = sys.argv.index("--path-out") + 1
    
    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]

    feature_engineering(input_path, output_path)

