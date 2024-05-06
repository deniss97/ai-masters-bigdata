import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import joblib
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder.appName("Prediction").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Читаем данные и обрабатываем отзывы непосредственно в Spark
    df_test = spark.read.json(test_data)

    # Загрузка модели
    pipeline = joblib.load(model_path)

    # Определяем UDF
    predict_udf = udf(lambda text: str(pipeline.predict([text])[0]), StringType())

    # Применяем модель к каждому отзыву
    df_result = df_test.withColumn('prediction', predict_udf(df_test['reviewText']))

    # Сохраняем результаты в CSV
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

