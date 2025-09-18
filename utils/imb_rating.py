from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, floor, when
import os
from dotenv import load_dotenv
from utils.user import User
import random

load_dotenv()

class IMDbRatingsConverter:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("IMDb Ratings Converter") \
            .config("spark.sql.adaptive.enabled", "true") \
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
    def load_data(self, min_votes: int):
        ratings = self.spark.read.jdbc(
            url=self.jdbc_url,
            table="title_ratings",
            properties=self.properties
            ).alias("r")

        titles = self.spark.read.jdbc(
            url=self.jdbc_url,
            table="title_basics",
            properties=self.properties
            ).alias("t")


        movies = (ratings.join(titles, ratings["tconst"] == titles["tconst"])
                  .filter(ratings["numVotes"] >= min_votes)
                  .select(
                      ratings["tconst"],
                      titles["primaryTitle"].alias("title"),
                      ratings["averageRating"],
                      ratings["numVotes"],
                      titles["genres"])
                  .orderBy(ratings["numVotes"].desc())
                  .limit(500))


        movies_data = movies.collect()
        print(f"Found {len(movies_data)} suitable movies")
        return movies_data
    
    
    def create_users_from_imdb_ratings(self, num_users=20, min_votes=10000):
        
        print("Creating users from IMDb rating patterns...")
        movies_data = self.load_data(min_votes)
        print(movies_data)
        user = User()
        created_users = []
        
        user_types = [
            ("action_fan", "Action,Adventure,Thriller"),
            ("drama_lover", "Drama,Romance,Biography"), 
            ("comedy_fan", "Comedy,Family,Animation"),
            ("scifi_geek", "Sci-Fi,Fantasy,Horror"),
            ("crime_fan", "Drama,Crime,War"),
            ("casual_viewer", ""), 
        ]
        
        for i in range(num_users):
            user_type_idx = i % len(user_types)
            user_type, preferred_genres = user_types[user_type_idx]
            
            username = f"{user_type}_{i+1}"
            email = f"{username}@example.com"
            
            user_id = user.create_user(username, email)
            if user_id:
                created_users.append({
                    'user_id': user_id,
                    'username': username,
                    'preferred_genres': preferred_genres.split(",") if preferred_genres else []
                })
        
        print(f"Created {len(created_users)} users")
        
        
        for user_info in created_users:
            user_id = user_info['user_id']
            preferred_genres = user_info['preferred_genres']
            
            num_ratings = random.randint(24, 56)
            if preferred_genres:
                suitable_movies = []
                for movie in movies_data:
                    genres = [(g or "").strip().lower() for g in (movie["genres"] or "").split(",")]
                    if any(genre.lower() in genres for genre in preferred_genres):
                        suitable_movies.append(movie)
            else:
                suitable_movies = movies_data
            
            movies_to_rate = random.sample(suitable_movies, min(num_ratings, len(suitable_movies)))
            
            for movie in movies_to_rate:
                imdb_rating = movie['averageRating']
                if imdb_rating >= 8.0:
                    user_rating = float(random.choices([3, 4, 5], weights=[10, 40, 50])[0])
                elif imdb_rating >= 7.0:
                    user_rating = float(random.choices([2, 3, 4, 5], weights=[5, 35, 45, 15])[0])
                elif imdb_rating >= 6.0:
                    user_rating = float(random.choices([1, 2, 3, 4], weights=[5, 30, 40, 25])[0])
                else:
                    user_rating = float(random.choices([1, 2, 3], weights=[40, 40, 20])[0])
                
            
                user.add_user_rating(user_id, movie['tconst'], user_rating)
            
            print(f"Added {len(movies_to_rate)} ratings for {user_info['username']}")
        
        self.spark.stop()
        
        print(f"Successfully created {len(created_users)} users with ratings based on IMDb data!")
        return True
    
    def analyze_imdb_data(self):
        ratings_df = self.spark.read \
            .jdbc(url=self.jdbc_url, table="title_ratings", properties=self.properties)
        
        titles_df = self.spark.read \
            .jdbc(url=self.jdbc_url, table="title_basics", properties=self.properties)

        analysis = ratings_df.alias("r") \
            .join(titles_df.alias("t"), col("r.tconst") == col("t.tconst")) \
            .filter(col("t.titleType") == "movie")
        
        total_movies = analysis.count()
        high_vote_movies = analysis.filter(col("r.numVotes") >= 10000).count()
        highly_rated = analysis.filter(col("r.averageRating") >= 8.0).count()
        
        print("IMDb Data Analysis:")
        print(f"Total movies: {total_movies:,}")
        print(f"Movies with 10K+ votes: {high_vote_movies:,}")
        print(f"Movies rated 8.0+: {highly_rated:,}")
        
        sample_movies = analysis.select(
            col("t.primaryTitle").alias("title"),
            col("r.averageRating"),
            col("r.numVotes")
        ).orderBy(col("numVotes").desc()).limit(10)
        
        print("\nTop 10 most voted movies:")
        for movie in sample_movies.collect():
            print(f"  • {movie['title']}: {movie['averageRating']:.1f} ({movie['numVotes']:,} votes)")
        
if __name__ == "__main__":
    converter = IMDbRatingsConverter()
    converter.analyze_imdb_data()
    converter.create_users_from_imdb_ratings()
    