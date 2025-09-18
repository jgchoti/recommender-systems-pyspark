from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import re

load_dotenv()

class User:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("User Movie Recommender") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.5") \
            .config("spark.driver.memory", "4g") \
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
        
        self.als_model = None
        self.setup_schemas()
    
    def setup_schemas(self):
        self.user_schema = StructType([
            StructField("userId", StringType(), False),
            StructField("username", StringType(), False),
            StructField("email", StringType(), True),
            StructField("createdAt", TimestampType(), True),
            StructField("isActive", BooleanType(), True)
        ])
        
        self.user_ratings_schema = StructType([
            StructField("userId", StringType(), False),
            StructField("tconst", StringType(), False),
            StructField("rating", FloatType(), False),
            StructField("ratedAt", TimestampType(), True)
        ])
        
        self.user_recommendations_schema = StructType([
            StructField("userId", StringType(), False),
            StructField("tconst", StringType(), False),
            StructField("predictedRating", FloatType(), False)
        ])
        
        self.movie_display_schema = StructType([
            StructField("tconst", StringType(), False),
            StructField("title", StringType(), False),
            StructField("genres", StringType(), True),
            StructField("year", IntegerType(), True),
            StructField("imdbRating", FloatType(), True),
            StructField("numVotes", IntegerType(), True)
        ])
    
    def setup_user_tables(self):
        print("🔧 Setting up user tables...")
        
        try:
            users_df = self.spark.createDataFrame([], self.user_schema)
            users_df.write \
                .option("createTableColumnTypes", 
                       "userId VARCHAR(50), username VARCHAR(100) NOT NULL, email VARCHAR(255), createdAt TIMESTAMP, isActive BOOLEAN") \
                .jdbc(url=self.jdbc_url, table="users", mode="append", properties=self.properties)
            

            ratings_df = self.spark.createDataFrame([], self.user_ratings_schema)
            ratings_df.write \
                .option("createTableColumnTypes", 
                       "userId VARCHAR(50), tconst VARCHAR(20), rating REAL, ratedAt TIMESTAMP") \
                .jdbc(url=self.jdbc_url, table="user_ratings", mode="append", properties=self.properties)
            

            recommendations_df = self.spark.createDataFrame([], self.user_recommendations_schema)
            recommendations_df.write \
                .option("createTableColumnTypes", 
                       "userId VARCHAR(50), tconst VARCHAR(20), predictedRating REAL") \
                .jdbc(url=self.jdbc_url, table="user_recommendations_generated", mode="append", properties=self.properties)
            
            print("✅ User tables created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up tables: {e}")
            return False
    
    def get_user_by_username(self, username):
        try:
            users_df = self.spark.read \
                .jdbc(url=self.jdbc_url, table="users", properties=self.properties, predicates=[f"username = '{username}'"])
            
            user = users_df.collect()
            
            if user:
                return user[0].asDict()
            else:
                return None
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
        
    def username_exists(self, username):
        try:
            users_df = self.spark.read \
                .jdbc(url=self.jdbc_url, table="users", properties=self.properties, predicates=[f"username = '{username}'"])
        
            return users_df.count() > 0
        except Exception as e:
            print(f"❌ Error checking username: {e}")
            return False

    def create_user(self, username, email=None):
        if not username or len(username.strip()) == 0:
            print("❌ Username cannot be empty")
            return None

        username = username.strip()

        if self.username_exists(username):
            print(f"❌ Username '{username}' is already taken")
            return None

        if not re.match("^[a-zA-Z0-9_.-]+$", username):
            print("❌ Username can only contain letters, numbers, underscores, dots, and hyphens")
            return None


        user_id = str(uuid.uuid4())
        user_data = [(user_id, username, email, datetime.now(), True)]

        try:
            new_user_df = self.spark.createDataFrame(user_data, self.user_schema)
            new_user_df.write \
                .jdbc(url=self.jdbc_url, table="users", mode="append", properties=self.properties)

            print(f"✅ User '{username}' created with ID: {user_id}")
            return user_id
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                print(f"❌ Username '{username}' is already taken")
            return None
        
    def add_user_rating(self, user_id, movie_id, user_rating):
        user_rating= [(user_id, movie_id, user_rating, datetime.now())]
        new_user_rating_df = self.spark.createDataFrame(user_rating, self.user_ratings_schema)
        new_user_rating_df.write \
                .jdbc(url=self.jdbc_url, table="user_ratings", mode="append", properties=self.properties)

        print(f"✅ User rating updated")

        
        

if __name__ == "__main__":       
    print("🔧 Setting up user tables...")
    user = User()
    user.setup_user_tables()
    print("👤 Creating demo user...")
    user_id = user.create_user("demo_user", "demo@example.com")
    user.create_user("ab") 
    
        