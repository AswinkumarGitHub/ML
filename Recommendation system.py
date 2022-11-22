# Databricks notebook source
# MAGIC %md 
# MAGIC #Installation of py$park

# COMMAND ----------

import pandas as pd

# COMMAND ----------

pip install pyspark

# COMMAND ----------


pip install findspark  

# COMMAND ----------

import os
# Install library for finding Spark
!pip install -q findspark
# Import the libary
import findspark
# Initiate findspark
findspark.init()
# Check the location for Spark
findspark.find()

# COMMAND ----------

# Import SparkSession
from pyspark.sql import SparkSession
# Create a Spark Session
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Check Spark Session Information
spark

# COMMAND ----------

# MAGIC %md 
# MAGIC #Importing the Pyspark modules and Creating the Session

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS

#Creating a Spark session
appName = "Recommender system in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md 
# MAGIC #Reading the CSV file of different users ratings and movies 

# COMMAND ----------

dbutils.fs.ls("/FileStore/tables/ratings.csv")

file_location = "/FileStore/tables/ratings.csv"
file_location1= "/FileStore/tables/movies.csv"
file_type = "csv"

ratings = spark.read.csv(file_location, inferSchema=True, header=True)
movies = spark.read.csv(file_location1, inferSchema=True, header=True)

#merge "movies" and "ratings" dataFrame based on "movieId"
rec = ratings.join(movies, "movieId")
rec.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #Data Preparation

# COMMAND ----------


#use only column data of "userId", "movieId", dan "rating"

data = ratings.select("userId", "movieId", "rating")

#Spliting the train and test data sets 

splits = data.randomSplit([0.7, 0.3])
train = splits[0].withColumnRenamed("rating", "label")
test = splits[1].withColumnRenamed("rating", "trueLabel")


#To print no.of rows of Traing and test data to show how its split 

train_rows = train.count()
test_rows = test.count()
print ("number of training data rows:", train_rows, 
       ", number of testing data rows:", test_rows)

# COMMAND ----------

# MAGIC %md 
# MAGIC #Model training 

# COMMAND ----------

#Define ALS (Alternating Least Square) as our recommender system

als = ALS(maxIter=19, regParam=0.01, userCol="userId", 
          itemCol="movieId", ratingCol="label")

#Train our ALS model

model = als.fit(train)
print("Training is done!")

# COMMAND ----------

# MAGIC %md 
# MAGIC #Predictions 

# COMMAND ----------

prediction = model.transform(test)
prediction.join(movies, "movieId").select(
    "userId", "title", "prediction", "trueLabel").show(n=10, truncate=False)
print("testing is done!")
#The prediction results The predictions and trueLabel are pretty closer

# COMMAND ----------



# COMMAND ----------

# MAGIC   %md 
# MAGIC   Error in our model 

# COMMAND ----------

#import RegressionEvaluator since we also want to calculate RMSE (Root Mean Square Error)

from pyspark.ml.evaluation import RegressionEvaluator

#To evaluate out prediction 

evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

#Nan - Not a numberical so we gonna drop those Nan

prediction.count()
a = prediction.count()
print("number of original data rows: ", a)
#drop rows with any missing data
cleanPred = prediction.dropna(how="any", subset=["prediction"])
b = cleanPred.count()
print("number of rows after dropping data with missing value: ", b)
print("number of missing data: ", a-b)

# COMMAND ----------

rmse = evaluator.evaluate(cleanPred)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

recs= model.recommendForAllUsers(10)
recs.show()
recs=recs.select("userId","recommendations.movieId") 
users = recs.select("userId").toPandas().iloc[0,0]
movies= recs.select("movieId").toPandas().iloc[0,0]

ratings_matrix=pd.DataFrame(movies,columns=["movieId"])
ratings_matrix["userId"] = users


ratings_matrix_ps = sqlContext.createDataFrame(ratings_matrix)
ratings_matrix_ps.show()

