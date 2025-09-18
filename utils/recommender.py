from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc, explode, split, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.linalg import Vectors   
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import explode, col, udf
from pyspark.sql.types import StringType
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class Recommender():
    def __init__(self, app_name= "recommendation_system"):
        self.app_name = app_name
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.5") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.DB_HOST = "localhost"
        self.DB_USER = os.getenv("DB_USER", "chotij")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        self.jdbc_url = f"jdbc:postgresql://{self.DB_HOST}:5432/imdb_recommendation"
        
        self.properties = {
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "driver": "org.postgresql.Driver"
        }
        self.model = None
        self.user_indexer = None
        self.movie_indexer = None
        self.model_path = "als_pipeline_model"
   
        
    def get_user_ratings_df(self):
        try:
            user_ratings_df = self.spark.read \
                .jdbc(url=self.jdbc_url, table="user_ratings", properties=self.properties)
            
            # print(ratings_df)
            if user_ratings_df:
                return user_ratings_df
            else:
                return None
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
        
    
    # (userId='e1990f19-64cb-4958-99c5-a00536754764', tconst='tt0373889', rating=3.0, ratedAt=datetime.datetime(2025, 9, 17, 11, 42, 29, 135889)
        
    def prepare_training_data(self, ratings_df):
        ratings = ratings_df.select(
            col("userId").cast("string"),
            col("tconst").cast("string"),
            col("rating").cast("float")
        )
        user_indexer = StringIndexer(inputCol="userId", outputCol="userIndexed", handleInvalid="skip")
        movie_indexer = StringIndexer(inputCol="tconst", outputCol="tconstIndexed", handleInvalid="skip")
        
        als = ALS(
            userCol="userIndexed",
            itemCol="tconstIndexed",
            ratingCol="rating",
            nonnegative=True,
            implicitPrefs=False,
            coldStartStrategy="drop"
        )
        
        pipeline = Pipeline(stages=[user_indexer, movie_indexer, als])

        return ratings, pipeline

    
    def train_model(self):
        print("🔄 Training new model...")
        user_ratings_df = self.get_user_ratings_df()
        total_ratings = user_ratings_df.count()
        unique_users = user_ratings_df.select("userId").distinct().count()
        unique_movies = user_ratings_df.select("tconst").distinct().count()
    
        print(f"📊 Raw data: {total_ratings} ratings, {unique_users} users, {unique_movies} movies")
        print("📈 User rating counts (top 10):")
        user_counts = user_ratings_df.groupBy("userId").count().orderBy(desc("count"))
        user_counts.show(10, False)
    
        ratings, pipeline = self.prepare_training_data(user_ratings_df)
        prep_total = ratings.count()
        prep_users = ratings.select("userId").distinct().count()
        print(f"📊 After preparation: {prep_total} ratings, {prep_users} users")
    
        train_df, test_df = ratings.randomSplit([0.8, 0.2], seed=42)
        als_stage = pipeline.getStages()[2]
        param_grid = ParamGridBuilder() \
        .addGrid(als_stage.rank, [200]) \
        .addGrid(als_stage.maxIter, [40]) \
        .addGrid(als_stage.regParam, [0.4]) \
        .build()

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )
    
        print("🏋️ Starting model training...")
        model = cv.fit(train_df)
        best_model = model.bestModel  
    
    
        user_indexer = best_model.stages[0]
        movie_indexer = best_model.stages[1]
    
        print(f"✅ Model trained with {len(user_indexer.labels)} users and {len(movie_indexer.labels)} movies")
        print("Sample included users:", user_indexer.labels[:5])
    
        best_model.write().overwrite().save(self.model_path) 
        self.model = best_model
    
        test_predictions = best_model.transform(test_df)
        rmse = evaluator.evaluate(test_predictions)
        print(f"✅ RMSE on test set: {rmse:.4f}")

        return best_model
        
    def generate_user_recommendations(self, num_recommendations=10):
        if self.model is None:
            self.train_model()
        als_model = self.model.stages[2]
        user_recs = als_model.recommendForAllUsers(num_recommendations)
        return user_recs
    
    def get_detailed_recommendations(self, user_recs):
    # Explode the recommendations array
        user_recs_exploded = user_recs.select(
            "userIndexed",
            explode("recommendations").alias("rec")
            ).select(
                "userIndexed",
                col("rec.tconstIndexed").alias("tconstIndexed"),
                col("rec.rating").alias("predicted_rating")
                )

   
        user_labels = self.model.stages[0].labels  
        movie_labels = self.model.stages[1].labels  


        user_index_to_string_udf = udf(lambda idx: user_labels[int(idx)], StringType())
        movie_index_to_string_udf = udf(lambda idx: movie_labels[int(idx)], StringType())


        user_recs_with_original = user_recs_exploded \
            .withColumn("userId_original", user_index_to_string_udf(col("userIndexed"))) \
            .withColumn("tconst_original", movie_index_to_string_udf(col("tconstIndexed")))

        # user_recs_with_original.select(
        # "userId_original",
        # "tconst_original",
        # "predicted_rating"
        # ).show(10, truncate=False)

        return user_recs_with_original
    
    def save_recommendations_to_db(self, user_recs_with_original):
        new_recommendations = user_recs_with_original.select(
        col("userId_original").alias("userId"),
        col("tconst_original").alias("tconst"),
        col("predicted_rating").alias("predictedRating")
        )

        try:
            new_recommendations.write \
                .jdbc(url=self.jdbc_url, table="user_recommendations_generated", mode="overwrite", properties=self.properties)

        except Exception as e:
            print(f"❌ Error saving recommendations to db: {e}")
            return None
    
    def get_saved_recommendations(self, user_id):
        try:
            rec_df = self.spark.read.jdbc(
                url=self.jdbc_url,
                table="user_recommendations_generated",
                properties=self.properties
            )

       
            movies_df = self.spark.read.jdbc(
                url=self.jdbc_url,
                table="title_basics",
                properties=self.properties
            )

       
            user_recs = rec_df.filter(rec_df.userId == user_id)
            joined_df = user_recs.join(movies_df, on="tconst", how="left")
            recommendations = [
                {
                    "tconst": row.tconst,
                    "title": row.primaryTitle,
                    "predictedRating": row.predictedRating,
                    "genres": row.genres,
                    "year": row.startYear
                }
                for row in joined_df.collect()
                ]

            return recommendations
        
        except Exception as e:
            print(f"❌ Error getting recommendations from db: {e}")
            return None
        
    # def content_based_recommendations(self, user_id, num_recommendations=10):

    def stop_spark(self):
        if self.spark:
            try:
                self.spark.stop()
            except Exception as e:
                print(f"⚠️ Spark already stopped or JVM gone: {e}")
        

# For testing
if __name__ == "__main__":
    recommender = Recommender()
    
    try:
        print("Testing movie recommender...")
      
        user_recs = recommender.generate_user_recommendations()
        user_recs_with_original = recommender.get_detailed_recommendations(user_recs)
        recommender.save_recommendations_to_db(user_recs_with_original)
        recs = recommender.get_saved_recommendations("421d59ca-932f-4806-9ce3-2076c205e87c")
        if recs:
            print("Top recommendations:")
            for rec in recs:
                print(f"  • {rec['title']} - Predicted: {rec['predictedRating']:.2f}/5")

    except Exception as e:
        print(f"Test error: {e}")
    finally:
        recommender.stop_spark()