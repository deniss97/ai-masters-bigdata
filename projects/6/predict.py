import sys
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, regexp_replace, lit
from pyspark.sql.types import LongType, DoubleType

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder \
        .appName("Prediction") \
        .config("spark.executor.memoryOverhead", "512m") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Чтение и предобработка данных
    df_test = spark.read.json(test_data)
    print("Схема данных после загрузки:")
    df_test.printSchema()

    # Проверка и обработка столбца 'vote'
    df_test = df_test.withColumn("vote", regexp_replace("vote", ",", "").cast(DoubleType()))
    df_test = df_test.fillna({'vote': 0.0})

    # Загрузка модели
    model = joblib.load(model_path)

    @pandas_udf("double", PandasUDFType.SCALAR)
    def predict_udf(votes):
        return model.predict(votes.values.reshape(-1, 1))

    # Применение модели к данным
    df_test = df_test.withColumn("prediction", predict_udf(df_test["vote"]))
    df_test = df_test.withColumn("id", df_test["vote"].cast(LongType()).alias("id"))

    # Преобразование результата предсказаний и сохранение
    df_test.select("id", "prediction").coalesce(1).write.csv(output_path, mode="overwrite", header=False)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

