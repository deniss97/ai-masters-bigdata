import sys
import numpy as np
import joblib
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import FloatType, StringType
from pyspark.sql import SparkSession, functions as F

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder.appName("Prediction").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Чтение и предобработка данных
    df_test = spark.read.json(test_data)
    print("Схема данных после загрузки:")
    df_test.printSchema()  # Для проверки доступных столбцов

    # Проверка и превращение данных, если 'vote' присутствует, если нет, создание 'vote' с типом float
    if 'vote' in df_test.columns:
        df_test = df_test.withColumn("vote", F.regexp_replace("vote", ",", "").cast(FloatType()))
    else:
        df_test = df_test.withColumn("vote", F.lit("0").cast(FloatType()))

    df_test = df_test.fillna({'vote': 0})  # Применение fillna на уровне DataFrame

    # Загрузка модели
    model = joblib.load(model_path)

    # Применение модели для каждой строки DataFrame
    pd_test = df_test.select("vote").toPandas()
    # Убедитесь, что данные имеют правильный тип
    pd_test['vote'] = pd_test['vote'].astype(float)
    predictions = model.predict(pd_test)

    # Обработка NaN в предсказаниях
    predictions = np.where(np.isnan(predictions), 0.5, predictions)

    # Преобразование списка предсказаний в Spark DataFrame
    # Предполагаем создание или использование идентификационного поля для объединения данных
    df_predictions = spark.createDataFrame(zip(list(range(len(predictions))), predictions), schema=["id", "prediction"]).cast(StringType())
    df_predictions = df_predictions.withColumn("id", F.col("id").cast(StringType()))

    # Ограничение на 4,000,000 записей
    df_predictions = df_predictions.limit(4000000)

    # Запись результатов без заголовков и индексов
    df_predictions.coalesce(1).write.mode('overwrite').option("header", "false").json(output_path)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1

    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

