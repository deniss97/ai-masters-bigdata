import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import joblib
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import numpy as np
from pyspark.sql.functions import regexp_replace

def predict(test_data, output_path, model_path):
    spark = SparkSession.builder.appName("Prediction").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df_test = spark.read.json(test_data)
    df_test = df_test.withColumn("vote", regexp_replace("vote", ",", "").cast(FloatType())).fillna({'vote': 0})
    
    model = joblib.load(model_path)
    
    # Применение модели для каждой строки DataFrame
    pd_test = df_test.select("vote").toPandas()
    predictions = model.predict(pd_test)
    
    # Запись результатов
    result_df = spark.createDataFrame(pd.DataFrame(predictions, columns=['prediction']))
    result_df.write.csv(output_path, mode='overwrite', header=True)

    spark.stop()
    
if __name__ == "__main__":
    input_arg_index = sys.argv.index("--test-in") + 1
    output_arg_index = sys.argv.index("--pred-out") + 1
    model_arg_index = sys.argv.index("--sklearn-model-in") + 1
    
    input_path = sys.argv[input_arg_index]
    output_path = sys.argv[output_arg_index]
    model_path = sys.argv[model_arg_index]

    predict(input_path, output_path, model_path)

