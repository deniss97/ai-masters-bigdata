import sys
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
import pandas as pd

def predict_task(input_path, output_path, model_path):
    # Создание Spark сессии
    spark = SparkSession.builder.appName("PredictionTask").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Определение схемы данных
    schema = StructType([
        StructField("id", StringType()),
        StructField("pred", FloatType())
    ])

    # Чтение и предобработка данных с использованием заданной схемы
    df_test = spark.read.schema(schema).json(input_path)
    df_test.cache()  # Кэширование DataFrame

    # Загрузка модели
    model = joblib.load(model_path)

    # Определение pandas_udf для применения модели
    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def prediction_udf(x: pd.Series) -> pd.Series:
        return pd.Series(model.predict(x.values.reshape(-1, 1)))

    # Применение pandas_udf к DataFrame и сохранение результатов
    df_test = df_test.withColumn("prediction", prediction_udf(df_test["pred"]))
    df_test = df_test.select("id", "prediction")

    # Сохранение данных в CSV файл без заголовков в указанном пути
    df_test.write.csv(output_path, mode='overwrite', header=False)

    # Завершение сессии Spark
    spark.stop()

if __name__ == "__main__":
    if "main" in globals():
        input_arg_index = sys.argv.index("--test-in") + 1
        output_arg_index = sys.argv.index("--pred-out") + 1
        model_arg_index = sys.argv.index("--sklearn-model-in") + 1

        input_path = sys.argv[input_arg_index]
        output_path = sys.argv[output_arg_index]
        model_path = sys.argv[model_arg_index]

        predict_task(input_path, output_path, model_path)

