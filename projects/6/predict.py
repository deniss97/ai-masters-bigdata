import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import joblib
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import numpy as np

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder.appName("Prediction").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df_test = spark.read.json(test_data)

    pipeline = joblib.load(model_path)

    # Создаём UDF, которая адаптирует данные под ожидания sklearn: преобразует список (1D) в 2D массив
    def model_predict(text):
        text_array = np.array([text]).reshape(1, -1)  # Преобразование в 2D массив
        return str(pipeline.predict(text_array)[0])

    predict_udf = udf(model_predict, StringType())

    # Применяем UDF к DataFrame
    df_result = df_test.withColumn('prediction', predict_udf(df_test['reviewText']))

    # Сохранение прогноза в CSV файл
    df_result.select('prediction').write.csv(output_path, mode='overwrite', header=True)

    spark.stop()
    
if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1
    
    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

