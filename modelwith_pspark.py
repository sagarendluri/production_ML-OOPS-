import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession \
 .builder \
 .appName("Python Spark SQL basic example") \
 .config("spark.some.config.option", "some-value") \
 .getOrCreate()
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
df2 = spark.read.csv(r"C:\Users\sagar\Downloads\top_5k_319_samples.csv",header  = True)
from pyspark.sql.functions import *
## Unwanted featues will be dropped
dfd = df.drop('_c0','index','AC','GT','GC','PV','TV','SB','PsT','LO','Line')
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
###  string columns converts into int or float
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dfd) for column in ["1st Layer Clusters"]]
## i am droping exsting column and new columns named with "_index"
pipeline = Pipeline(stages=indexers)
ddf = pipeline.fit(dfd).transform(dfd)
ddf = ddf.drop("1st Layer Clusters")
final = ddf.select([col(c).cast('int') for c in ddf.columns])
# final.printSchema()
feature = VectorAssembler(inputCols=final.columns[1:],outputCol="features")
feature_vector= feature.transform(final)
(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="1st Layer Clusters_index", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "1st Layer Clusters_index", "features").show()

# trainingData