from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer
from pyspark.ml.regression import LinearRegression

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
indexer = StringIndexer(inputCol="overall", outputCol="label")
lr = LinearRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[tokenizer, hashingTF, indexer, lr])

