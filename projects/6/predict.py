import sys
import numpy as np
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder.appName("DataCopying").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Чтение и предобработка данных
    df_test = spark.read.json(test_data)
    # df_test = df_test.withColumn("vote", regexp_replace("vote", ",", "").cast(FloatType())).fillna({'vote': 0})

    # Ограничение на количество записей задаем здесь на всякий случай
    df_test.limit(4000000).coalesce(1).write.mode('overwrite').option("header", "true").csv(output_path)

    # Завершение сессии Spark
    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

