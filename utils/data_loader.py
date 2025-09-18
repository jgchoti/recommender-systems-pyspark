from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import os
from dotenv import load_dotenv
import time

class DataLoader:
    def __init__(self, data_path="data"):
        load_dotenv()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_dir, data_path)
        
        DB_HOST = os.getenv("DB_HOST")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        self.jdbc_url = f"jdbc:postgresql://localhost:5432/imdb_recommendation"

        self.properties = {
            "user": DB_USER,
            "password": DB_PASSWORD,
            "driver": "org.postgresql.Driver",
            "batchsize": "10000", 
            "numPartitions": "4",   
            "rewriteBatchedStatements": "true",
            "prepStmtCacheSize": "250",
            "prepStmtCacheSqlLimit": "2048"
        }

        self.spark = SparkSession.builder \
            .appName("IMDb Recommendation") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.5") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "20") \
            .getOrCreate()

        self.titles_schema = StructType([
            StructField("tconst", StringType()),
            StructField("titleType", StringType()),
            StructField("primaryTitle", StringType()),
            StructField("originalTitle", StringType()),
            StructField("isAdult", IntegerType()),
            StructField("startYear", StringType()),     
            StructField("endYear", StringType()),      
            StructField("runtimeMinutes", StringType()),
            StructField("genres", StringType())
        ])

        self.akas_schema = StructType([
            StructField("titleId", StringType()),
            StructField("ordering", IntegerType()),
            StructField("title", StringType()),
            StructField("region", StringType()),
            StructField("language", StringType()),
            StructField("types", StringType()),       
            StructField("attributes", StringType()),   
            StructField("isOriginalTitle", IntegerType())
        ])

        self.ratings_schema = StructType([
            StructField("tconst", StringType()),
            StructField("averageRating", FloatType()),
            StructField("numVotes", IntegerType())
        ])

        self.principals_schema = StructType([
            StructField("tconst", StringType()),
            StructField("ordering", IntegerType()),
            StructField("nconst", StringType()),
            StructField("category", StringType()),
            StructField("job", StringType()),
            StructField("characters", StringType())
        ])

        self.episodes_schema = StructType([
            StructField("tconst", StringType()),
            StructField("parentTconst", StringType()),
            StructField("seasonNumber", StringType()),   
            StructField("episodeNumber", StringType())   
        ])

        self.crew_schema = StructType([
            StructField("tconst", StringType()),
            StructField("directors", StringType()), 
            StructField("writers", StringType())     
        ])

        self.names_schema = StructType([
            StructField("nconst", StringType()),
            StructField("primaryName", StringType()),
            StructField("birthYear", StringType()),      
            StructField("deathYear", StringType()),     
            StructField("primaryProfession", StringType()), 
            StructField("knownForTitles", StringType())     
        ])

        self.user_recommendations_schema = StructType([
            StructField("userId", StringType()),
            StructField("tconst", StringType()),
            StructField("predictedRating", FloatType())
        ])

        self.schemas = {
            "title.basics.tsv.gz": self.titles_schema,
            "title.akas.tsv.gz": self.akas_schema,
            "title.ratings.tsv.gz": self.ratings_schema,
            "title.principals.tsv.gz": self.principals_schema,
            "title.episode.tsv.gz": self.episodes_schema,
            "title.crew.tsv.gz": self.crew_schema,
            "name.basics.tsv.gz": self.names_schema,
        }
    
    def load_data(self):
        for i, (file_name, schema) in enumerate(self.schemas.items(), 1):
            try:
                start_time = time.time()
                full_path = os.path.join(self.data_path, file_name)
                print(f"[{i}/{len(self.schemas)}] Processing {file_name}...")

                df = self.spark.read.csv(
                    full_path,
                    sep="\t",
                    header=True,
                    schema=schema,
                    nullValue="\\N"
                )

                row_count = df.count()
                print(f"  - Rows: {row_count:,}")
                

                if row_count > 1000000:
                    df = df.repartition(4)
                elif row_count > 100000:
                    df = df.repartition(2)
                else:
                    df = df.coalesce(1)

                table_name = file_name.replace(".tsv.gz","").replace(".", "_").replace("-", "_")
                print(f"  - Writing to table {table_name}...")

                df.write \
                  .option("batchsize", 5000) \
                  .jdbc(url=self.jdbc_url, table=table_name, mode="overwrite", properties=self.properties)

                elapsed_time = time.time() - start_time
                print(f"  - Finished {table_name} in {elapsed_time:.1f} seconds")
                
                self.spark.catalog.clearCache()

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

        self.spark.stop()
        print("All files processed!")

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()