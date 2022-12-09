# import pyspark
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import os


conf = SparkConf().setAppName("Chashmmini-Spark").set("spark.executor.memory", "1024M").set(
    "spark.driver.memory", "1024M").set("spark.cores.max", "2").set(
        "spark.executor.cores", "2").set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.3.1')

SPARK_URL = None
if os.environ.get('SPARK_URL'):
    SPARK_URL = os.environ.get('SPARK_URL')
    print("Using Spark URL: {}".format(SPARK_URL))
    conf.setMaster(SPARK_URL)
else:
    print("Using local Spark")

sparkContext = SparkContext(conf=conf)

sparkContext.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')

configs = [("fs.s3a.access.key", "AKIAZMMBFGUYTD33RVOS"),
           ("fs.s3a.secret.key", "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"),
           ("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com"),
           ("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
           ]

hadoopConf = sparkContext._jsc.hadoopConfiguration()

for key, value in configs:
    hadoopConf.set(key, value)
    print(f"Setting {key} to {value}")

spark = SparkSession(sparkContext=sparkContext).builder.getOrCreate()


TRAINING_DF_LOCATION = "s3a://wine-data-spark/TrainingDataset.csv"
print(
    f"Downloading/Getting training data from S3 - ({TRAINING_DF_LOCATION})!!")
df = spark.read.csv(TRAINING_DF_LOCATION,
                    header=True, inferSchema=True, sep=";")

assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")

output = assembler.transform(df)

dt = DecisionTreeClassifier(labelCol="quality", featuresCol="features")

print("Building model from training data!!")
dtModel = dt.fit(output)


VALIDATION_DF_LOCATION = "s3a://wine-data-spark/ValidationDataset.csv"
print(f"Downloading/Getting validation data from S3({VALIDATION_DF_LOCATION})")
validation_df = spark.read.csv(
    VALIDATION_DF_LOCATION, header=True, inferSchema=True, sep=";")

validation_output = assembler.transform(validation_df)

print("Validating model!!")
predictions = dtModel.transform(validation_output)

evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = 1 - evaluator.evaluate(predictions)
accuracy = accuracy * 100
print(f"Accuracy: {accuracy:.3f}%")

print("Persisting model!!")
print("Pushing model to S3!!")
MODEL_LOCATION = "s3a://wine-data-spark/model"
dtModel.write().overwrite().save(MODEL_LOCATION)
