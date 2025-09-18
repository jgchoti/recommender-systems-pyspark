import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from utils.user import User

load_dotenv()

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'current_username' not in st.session_state:
    st.session_state.current_username = None
if 'user_system' not in st.session_state:
    st.session_state.user_system = None
if 'recommender_system' not in st.session_state:
    st.session_state.recommender_system = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_resource
def init_user_system():
    return User()

@st.cache_resource
def get_sql_engine():
    DB_USER = os.getenv("DB_USER", "chotij")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/imdb_recommendation'
    
    engine = create_engine(
        connection_string,
        pool_pre_ping=True,  
        pool_recycle=300,    
        echo=False           
    )
    return engine

def test_database_connection():
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return False

def table_exists_sqlalchemy(table_name: str) -> bool:
    """Check if a table exists using SQLAlchemy"""
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='public'
                  AND table_name=:t
                """),
                {"t": table_name},
            ).fetchone()
            return result is not None
    except Exception as e:
        print(f"Error checking table '{table_name}': {e}")
        return False

def table_exists(table_name: str):
    if table_exists_sqlalchemy(table_name):
        return True
    if st.session_state.get("user_system") is not None:
        try:
            spark = st.session_state.user_system.spark
            jdbc_url = st.session_state.user_system.jdbc_url
            props = st.session_state.user_system.properties
            query = f"(SELECT 1 FROM {table_name} LIMIT 1) AS tmp"
            spark.read.jdbc(url=jdbc_url, table=query, properties=props).collect()
            return True
        except Exception:
            return False
    else:
        return False

def table_non_empty(table_name: str):
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT 1 FROM "{table_name}" LIMIT 1')
            ).fetchone()
            return result is not None
    except Exception as e:
        print(f"Error checking non-empty for table '{table_name}': {e}")
        return False

def imdb_data_ready() -> bool:
    return table_non_empty("title_basics") and table_non_empty("title_ratings")

def user_tables_ready() -> bool:
    return (table_exists("users") and 
            table_exists("user_ratings") and 
            table_exists("user_recommendations_generated"))

def get_user_rating_stats_sql(user_id: str):
    try:
        engine = get_sql_engine()
        count_df = pd.read_sql(
            text('SELECT COUNT(*) AS cnt, AVG(rating) AS avg_rating FROM user_ratings WHERE "userId" = :uid'),
            engine,
            params={"uid": user_id},
        )
        cnt = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
        avg_rating = float(count_df.iloc[0]["avg_rating"]) if not pd.isna(count_df.iloc[0]["avg_rating"]) else None
        
        recent_df = pd.read_sql(
            text('''
            SELECT ur.tconst, ur.rating, ur."ratedAt", tb."primaryTitle" as title
            FROM user_ratings ur
            LEFT JOIN title_basics tb ON ur.tconst = tb.tconst
            WHERE ur."userId" = :uid
            ORDER BY ur."ratedAt" DESC
            LIMIT 5
            '''),
            engine,
            params={"uid": user_id},
        )
        return cnt, avg_rating, recent_df
    except Exception as e:
        print(f"SQL error, falling back to Spark: {e}")
        return get_user_rating_stats(user_id)

def get_saved_recommendations_sql(user_id: str):
    print("user_id:", user_id)
    try:
        engine = get_sql_engine()
        query = text('''
        SELECT r.tconst, b."primaryTitle" AS title, r."predictedRating", b.genres, b."startYear" AS year
        FROM user_recommendations_generated r
        LEFT JOIN title_basics b ON r.tconst = b.tconst
        WHERE r."userId" = :uid
        ORDER BY r."predictedRating" DESC
        LIMIT 50
        ''')
        df = pd.read_sql(query, engine, params={"uid": user_id})
        print("read rec", df)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"SQL error, falling back to Spark: {e}")
        return get_saved_recommendations(user_id)

def get_user_rating_stats(user_id: str):
    if st.session_state.user_system is not None:
        try:
            spark = st.session_state.user_system.spark
            props = st.session_state.user_system.properties
            jdbc_url = st.session_state.user_system.jdbc_url
            ratings_df = spark.read.jdbc(url=jdbc_url, table="user_ratings", properties=props)
            user_df = ratings_df.filter(ratings_df.userId == user_id)
            cnt = user_df.count()
            if cnt == 0:
                return 0, None, pd.DataFrame()
            from pyspark.sql.functions import avg as _avg, col, desc
            avg_row = user_df.agg(_avg(col("rating")).alias("avg_rating")).collect()[0]
            avg_rating = float(avg_row["avg_rating"]) if avg_row and avg_row["avg_rating"] is not None else None
            recent_rows = user_df.orderBy(desc("ratedAt")).limit(5).toPandas()
            return cnt, avg_rating, recent_rows
        except Exception:
            return 0, None, pd.DataFrame()
    return 0, None, pd.DataFrame()

def get_saved_recommendations(user_id: str):
    if st.session_state.user_system is not None:
        try:
            spark = st.session_state.user_system.spark
            props = st.session_state.user_system.properties
            jdbc_url = st.session_state.user_system.jdbc_url
            rec_df = spark.read.jdbc(url=jdbc_url, table="user_recommendations_generated", properties=props)
            basics_df = spark.read.jdbc(url=jdbc_url, table="title_basics", properties=props)
            user_recs = rec_df.filter(rec_df.userId == user_id)
            joined = user_recs.join(basics_df, on="tconst", how="left")
            rows = joined.select("tconst", basics_df.primaryTitle.alias("title"), "predictedRating", basics_df.genres, basics_df.startYear.alias("year")).collect()
            return [
                {
                    "tconst": r["tconst"],
                    "title": r["title"],
                    "predictedRating": r["predictedRating"],
                    "genres": r["genres"],
                    "year": r["year"],
                }
                for r in rows
            ]
        except Exception as e:
            print(f"Spark error: {e}")
            return []
    return []

def get_user_rated_tconsts(user_id: str) -> set:
    try:
        engine = get_sql_engine()
        df = pd.read_sql(
            text('SELECT tconst FROM user_ratings WHERE "userId" = :uid'),
            engine, 
            params={"uid": user_id}
        )
        return set(df['tconst'].tolist()) if not df.empty else set()
    except Exception as e:
        print(f"Error fetching rated tconsts for user {user_id}: {e}")
        return set()

@st.cache_data(ttl=300)
def get_popular_movies(limit=100):
    query = text(f"""
        SELECT
            b.tconst,
            b."primaryTitle" AS title,
            b.genres,
            CAST(b."startYear" AS INTEGER) AS year,
            r."averageRating" AS imdb_rating,
            r."numVotes" AS votes
        FROM title_basics b
        JOIN title_ratings r ON b.tconst = r.tconst
        WHERE b."titleType" = 'movie'
          AND b."isAdult" = '0'
          AND b."startYear" ~ '^[0-9]+$'
          AND CAST(b."startYear" AS INTEGER) >= 1980
          AND r."numVotes" >= 25000
          AND r."averageRating" >= 6.5
        ORDER BY r."numVotes" DESC
        LIMIT {limit}
    """)
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("*Rate movies and get personalized recommendations using PySpark ALS*")
    
    if not test_database_connection():
        st.error("❌ Cannot connect to database. Please check your PostgreSQL connection.")
        st.info("Your database appears to be running, but there might be credential issues. Check your .env file.")
        return
   
    if st.session_state.user_system is None:
        with st.spinner("Initializing system..."):
            try:
                st.session_state.user_system = init_user_system()
                if not user_tables_ready():
                    st.session_state.user_system.setup_user_tables()
            except Exception as e:
                st.error(f"❌ Failed to initialize user system: {e}")
                return
    
    st.sidebar.header("🔧 System Management")
    
    with st.sidebar.expander("📊 Data Status", expanded=False):
        imdb_ready = imdb_data_ready()
        st.write(f"IMDb data status: {'✅ Ready' if imdb_ready else '❌ Missing/Empty'}")
        users_ok = table_exists('users')
        ratings_ok = table_exists('user_ratings')
        recs_ok = table_exists('user_recommendations_generated')
        st.write(f"Users table: {'✅' if users_ok else '❌'} | Ratings: {'✅' if ratings_ok else '❌'} | Recs: {'✅' if recs_ok else '❌'}")
        if not imdb_ready:
            st.info("App will operate in limited mode until IMDb tables are populated.")

    st.sidebar.header("👤 User Management")
    if st.session_state.current_username == None:
        st.sidebar.subheader("Create User")
        with st.sidebar.form("create_user"):
            username = st.text_input("Username")
            email = st.text_input("Email (optional)")
            if st.form_submit_button("Create User"):
                if username:
                    try:
                        user_id = st.session_state.user_system.create_user(username, email)
                        if user_id:
                            st.session_state.current_user_id = user_id
                            st.session_state.current_username = username
                            st.sidebar.success(f"✅ User '{username}' created!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to create user")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error creating user: {e}")
                else:
                    st.sidebar.error("Please enter a username")

        st.sidebar.subheader("Login")
        with st.sidebar.form("login_user"):
            login_username = st.text_input("Enter username to login")
            if st.form_submit_button("Login"):
                if login_username:
                    try:
                        user_data = st.session_state.user_system.get_user_by_username(login_username)
                        if user_data:
                            st.session_state.current_user_id = user_data['userId']
                            st.session_state.current_username = user_data['username']
                            st.sidebar.success(f"✅ Logged in as {login_username}")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ User not found")
                    except Exception as e:
                        st.sidebar.error(f"❌ Login error: {e}")
                else:
                    st.sidebar.error("Please enter a username")

    if st.session_state.current_username:
        st.sidebar.info(f"👤 Current user: {st.session_state.current_username}")
        if st.sidebar.button("Logout"):
            st.session_state.current_user_id = None
            st.session_state.current_username = None
            st.rerun()
    
    if not st.session_state.current_user_id:
        st.info("👈 Please create a user or login to get started")
        
        st.subheader("🎬 Popular Movies in Database")
        movies_df = get_popular_movies(10)
        if not movies_df.empty:
            for _, movie in movies_df.iterrows():
                st.write(f"• **{movie['title']}** ({movie['year']}) - {movie['imdb_rating']:.1f}/10 ⭐")
        return
    
    tab1, tab2, tab3 = st.tabs(["⭐ Rate Movies", "🎬 Get Recommendations", "👤 My Profile"])
    
    with tab1:
        st.header("⭐ Rate Movies")
        st.info("Rate at least 10-15 movies to get good recommendations")
        
        movies_df = get_popular_movies()
        rated = get_user_rated_tconsts(st.session_state.current_user_id)
        if rated:
            movies_df = movies_df[~movies_df['tconst'].isin(rated)]
        
        if movies_df.empty:
            st.error("No movies found. Make sure IMDb data is loaded.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            all_genres = set()
            for genres in movies_df['genres'].dropna():
                if genres != '\\N' and pd.notna(genres):
                    all_genres.update([g.strip() for g in str(genres).split(',')])
            
            genre_filter = st.selectbox("Filter by genre:", ["All"] + sorted(all_genres))
        
        with col2:
            year_filter = st.slider("Minimum year:", 1980, 2023, 2000)
        
        filtered_movies = movies_df.copy()
        if genre_filter != "All":
            filtered_movies = filtered_movies[
                filtered_movies['genres'].str.contains(genre_filter, case=False, na=False)
            ]
        filtered_movies = filtered_movies[filtered_movies['year'] >= year_filter]
        
        st.subheader(f"Movies to Rate ({len(filtered_movies)} available)")

        for i, movie in filtered_movies.head(20).iterrows():
            with st.expander(f"{movie['title']} ({movie['year']}) - {movie['imdb_rating']:.1f}/10"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**IMDb Rating:** {movie['imdb_rating']:.1f}/10 ({movie['votes']:,} votes)")
                
                with col2:
                    # Rating selector
                    rating = st.selectbox(
                        "Your rating:",
                        options=[0, 1, 2, 3, 4, 5],
                        format_func=lambda x: "Not Rated" if x == 0 else f"{x}/5 ⭐",
                        key=f"rating_{movie['tconst']}"
                    )
                    
                    if rating > 0:
                        if st.button(f"Save Rating", key=f"save_{movie['tconst']}"):
                            try:
                                st.session_state.user_system.add_user_rating(
                                    st.session_state.current_user_id, 
                                    movie['tconst'], 
                                    float(rating)
                                )
                                st.success(f"✅ Rated {rating}/5")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to save rating: {e}")
    
    with tab2:
        st.header("🎬 Recommendations")
        min_ratings = 10
        cnt, _, _ = get_user_rating_stats_sql(st.session_state.current_user_id)
        if cnt < min_ratings:
            st.info(f"Rate at least {min_ratings} movies to unlock recommendations. You have rated {cnt}.")
        else:
            try:
                current_recs = get_saved_recommendations_sql(st.session_state.current_user_id)
            except Exception:
                current_recs = []

          
            do_refresh = st.button("🔁 Refresh Recommendations")
           
            if do_refresh:
                with st.spinner("Building model and generating recommendations..."):
                    try:
                        from utils.recommender import Recommender
                        recommender = Recommender()
                        recommender.train_model()
                        user_recs = recommender.generate_user_recommendations()
                        user_recs_with_original = recommender.get_detailed_recommendations(user_recs)
                        recommender.save_recommendations_to_db(user_recs_with_original)
                        current_recs = get_saved_recommendations(st.session_state.current_user_id)
                        st.success("✅ Recommendations updated")
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {e}")

            if current_recs:
                for i, rec in enumerate(current_recs[:10], 1):
                    with st.expander(f"{i}. {rec['title']} ({rec['year']})"):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.write(f"**Genres:** {rec['genres']}")
                            st.write(f"**Predicted Rating:** {rec['predictedRating']:.2f}/5 ⭐")
                        with c2:
                            st.metric("Confidence", f"{rec['predictedRating']:.1f}")
    
    with tab3:
        st.header("👤 My Profile")
        
        if st.session_state.current_username:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Username", st.session_state.current_username)
            with col2:
                st.metric("User ID", st.session_state.current_user_id[:8] + "...")
            with col3:
                cnt, avg_rating, _ = get_user_rating_stats_sql(st.session_state.current_user_id)
                st.metric("Movies Rated", cnt if cnt is not None else 0)
            
            st.subheader("📊 Your Rating Statistics")
            

            cnt, avg_rating, recent_df = get_user_rating_stats_sql(st.session_state.current_user_id)
            if cnt and cnt > 0:
                st.write(f"You have rated **{cnt}** movies")
                if avg_rating is not None:
                    st.metric("Average Rating Given", f"{avg_rating:.2f}/5")
                st.subheader("🎬 Your Recent Ratings")
                if not recent_df.empty:
                    for _, row in recent_df.iterrows():
                        movie_title = row.get('title', row['tconst'])
                        st.write(f"• **{movie_title}** - {row['rating']}/5 ⭐")
            else:
                st.info("No ratings yet. Start rating movies to see your statistics!")

if __name__ == "__main__":
    main()