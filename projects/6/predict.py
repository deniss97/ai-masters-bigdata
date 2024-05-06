import sys
import numpy as np
import joblib
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import regexp_replace, lit, col
from pyspark.sql.types import FloatType

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

    # Проверка наличия и обработка столбца 'vote'
    if 'vote' in df_test.columns:
        df_test = df_test.withColumn("vote", regexp_replace(col("vote"), ",", "").cast(FloatType()))
    else:
        df_test = df_test.withColumn("vote", lit(0.0).cast(FloatType()))

    # Заполнение пропусков
    df_test = df_test.fillna({'vote': 0.0})

    # Загрузка модели
    model = joblib.load(model_path)

    # Применение модели к признакам
    pd_test = df_test.select("vote").toPandas()
    predictions = model.predict(pd_test['vote'].values.reshape(-1, 1))

    # Обработка NaN в предсказаниях
    predictions = np.where(np.isnan(predictions), 0.5, predictions)

    # Преобразование результата предсказаний в Spark DataFrame
    rows = [Row(id=i, prediction=float(pred)) for i, pred in enumerate(predictions)]
    result_df = spark.createDataFrame(rows)

    # Ограничение на 4,000,000 записей
    result_df = result_df.limit(4000000)

    # Запись результатов
    result_df.coalesce(1).write.mode('overwrite').option("header", "false").json(output_path)

    spark.stop()

if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1
    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]
    predict(input_path, output_path, model_path)

