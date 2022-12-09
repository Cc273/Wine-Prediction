from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import os


conf = SparkConf().setAppName("Chashmmini-Spark").set("spark.executor.memory", "1024M").set(
    "spark.driver.memory", "1024M").set("spark.cores.max", "2").set(
        "spark.executor.cores", "2").set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.3.1')

SPARK_URL = None
if os.environ.get('SPARK_URL'):
    SPARK_URL = os.environ.get('SPARK_URL')
    print(f"Using Spark URL: {SPARK_URL}")
    conf.setMaster(SPARK_URL)
else:
    print("Using local Spark")

sparkContext = SparkContext(conf=conf)

configs = [("fs.s3a.access.key", "AKIAZMMBFGUYTD33RVOS"),
           ("fs.s3a.secret.key", "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"),
           ("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com"),
           ("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
           ]

hadoopConf = sparkContext._jsc.hadoopConfiguration()

for key, value in configs:
    hadoopConf.set(key, value)
    print(f"Setting {key} to {value}")

# Enable S3 V4 signature support as required by AWS
sparkContext.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')


spark = SparkSession(sparkContext=sparkContext).builder.getOrCreate()

MODEL_LOCATION = "s3a://wine-data-spark/model"
print(f"Downloading model from s3 bucket: {MODEL_LOCATION}")


# Load model from S3
dtModel = DecisionTreeClassificationModel.load(MODEL_LOCATION)

VALIDATION_DF_LOCATION = "s3a://wine-data-spark/ValidationDataset.csv"
print(
    f"Downloading validation dataset from s3 bucket: {VALIDATION_DF_LOCATION}")
validation_df = spark.read.csv(
    VALIDATION_DF_LOCATION, header=True, inferSchema=True, sep=";")


assembler = VectorAssembler(
    inputCols=validation_df.columns[:-1], outputCol="features")

validation_output = assembler.transform(validation_df)

predictions = dtModel.transform(validation_output)

evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")

accuracy = 1 - evaluator.evaluate(predictions)
print(f"Prediction Accuracy: {accuracy * 100 :.3f}%")
