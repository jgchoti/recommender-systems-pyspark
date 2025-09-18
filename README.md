# Recommendation System using PySpark's ALS Algorithm

Using PySpark’s ALS algorithm to deliver personalized movie recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![ALS Algorithm](https://img.shields.io/badge/ML-ALS%20Collaborative%20Filtering-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## Project Overview

A PySpark-powered movie recommendation system built on collaborative filtering principles. Using ALS (Alternating Least Squares), the system identifies users with similar tastes and recommends titles based on their preferences.

Key capabilities:

- Process 11,907,778 titles and millions of ratings efficiently
- Discover hidden taste communities with ALS latent factors
- Deliver personalized recommendations through a Streamlit interface

---

## The Story Behind This Project

People naturally gravitate toward others with shared interests—and movie preferences are no exception. This project uses ALS to detect hidden taste communities in IMDb rating data, connecting users with similar preferences to deliver meaningful recommendations.

What I explored:

- **Collaborative Filtering**: How users naturally cluster based on rating patterns
- **ALS Algorithm Mechanics**: Extracting latent factors that represent taste communities
- **Scalable Pattern Detection**: Processing millions of ratings to identify communities at scale

---

Features

- **Data Ingestion & Cleaning**: Load, parse, and clean 7 IMDb datasets into PostgreSQL, ensuring high-quality structured data
- **User Management System**: Collect, validate, and manage user ratings to build individual taste profiles.
- **Collaborative Filtering Engine**: Use ALS (Alternating Least Squares) to identify latent taste communities and generate personalized recommendations.
- **Synthetic Rating Generator**: Convert IMDb aggregated ratings into realistic user rating profiles for model training.
- **Interactive Web App**: Streamlit interface for users to rate titles and explore personalized recommendations in real time.

---

## Architecture

```
📦 Movie Recommender
├── 🗃️  DataLoader Class         # Load & clean IMDb datasets
├── 👤  User Class               # Manage taste profiles & ratings
├── 🤖  MovieRecommender         # ALS collaborative filtering & predictions
├── 🔄  IMDb Rating Converter    # Generate realistic synthetic ratings
└── 🌐  Streamlit App            # Interactive recommendation interface
```

---

## Tech Stack

- **PySpark**: Distributed computing for large-scale ML model training
- **ALS Algorithm**: Matrix factorization for collaborative filtering
- **PostgreSQL**: Robust storage for imb data, users, and ratings
- **Streamlit**: Interactive web interface for rating and recommendations
- **Pandas**: Data analysis and community pattern exploration

---

## Quick Start

### Prerequisites

- Python 3.9+
- PySpark 3.5.0+
- PostgreSQL 14+
- Java 8+ (required for PySpark)

### Installation

1. **Clone the repository**

```bash
git clone git@github.com:jgchoti/recommender-systems-pyspark.git
cd recommender-systems-pyspark
```

2. **Set up virtual environment**

```bash
python -m venv env
source env/bin/activate
```

3. **Install dependencies**

```bash
pip install pyspark pandas streamlit sqlalchemy psycopg2-binary python-dotenv
```

4. **Set up environment variables**
   Create a `.env` file in the project root:

```env
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password
DB_HOST=localhost
```

5. **Set up PostgreSQL database**

```sql
CREATE DATABASE imdb_recommendation;
```

### Running the Application

1. **Load IMDb data** (first time only)

```bash
python utils/data_loader.py
```

2. **Set up user system**

```bash
python utils/user.py
```

3. **Generate sample users with ratings** (optional)

```bash
python utils/imb_rating.py
```

4. **Launch the web application**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## How It Works

### ALS Algorithm: Finding Taste Communities

ALS discovers latent factors that naturally cluster users with similar preferences:

- **Latent Factor Discovery**: Identifies hidden taste dimensions
- **Natural Clustering**: Groups users with similar rating patterns
- **Community-Based Recommendations**: Recommends titles highly rated by other users in the same taste cluster.

### Smart Training Data from IMDb Patterns

IMDb’s aggregated ratings are converted into realistic individual user preferences:

- Titles rated 8.0+ → Attract users rating 4–5 stars
- Titles rated 7.0–7.9 → Users rating 3–4 stars
- Titles rated 6.0–6.9 → Users rating 2–4 stars

This ensures training data reflects authentic taste communities.

### Recommendation Pipeline

1. User rates Titles through the Streamlit interface
1. ALS identifies their taste community
1. System finds Titles favored by similar users
1. Delivers personalized recommendations with confidence scores

---

## Learning Highlights

- **Collaborative Filtering Psychology**: Users with similar tastes cluster naturally in mathematical space
- **ALS Algorithm Elegance**: Latent factors emerge to represent genuine taste communities
- **Community-Driven Insights**: Individual preferences become meaningful within taste communities
- **Scale & Pattern Recognition**: Large-scale processing reveals subtle preference signals

---

### Application Features

The integrated Streamlit application (`app.py`) provides:

1. **🔧 System Management**

   - Load IMDb data into PostgreSQL
   - Generate sample users with realistic ratings
   - Database connection testing

2. **👤 User Management**

   - Create new user accounts
   - Login/logout functionality
   - User profile management

3. **⭐ Movie Rating Interface**

   - Browse popular movies from IMDb
   - Filter by genre and year
   - Rate movies on a 1-5 star scale
   - Real-time rating submission

4. **🎬 Recommendation Engine**

   - Train ALS collaborative filtering model
   - Generate personalized recommendations
   - View recommendations with confidence scores
   - Display movie details and predicted ratings

5. **📊 User Profile & Statistics**
   - View rating history
   - Track average ratings given
   - Monitor recommendation performance

---

## Sample Dashboard

### Application Features

The integrated Streamlit application (`app.py`) provides:

1. **🔧 System Management**

   - Load IMDb data into PostgreSQL
   - Generate sample users with realistic ratings
   - Database connection testing

2. **👤 User Management**

   - Create new user accounts
   - Login/logout functionality
   - User profile management
     ![app_screenshot](/assets/demo-1.png)

3. **⭐ Movie Rating Interface**

   - Browse popular movies from IMDb
   - Filter by genre and year
   - Rate movies on a 1-5 star scale
   - Real-time rating submission
     ![app_screenshot](/assets/demo-2.png)

4. **🎬 Recommendation Engine**

   - Train ALS collaborative filtering model
   - Generate personalized recommendations
   - View recommendations with confidence scores
   - Display movie details and predicted ratings
     ![app_screenshot](/assets/demo-3.png)

5. **📊 User Profile & Statistics**
   - View rating history
   - Track average ratings given
   - Monitor recommendation performance
     ![app_screenshot](/assets/demo-4.png)

---

## Future Improvements

- Real-time recommendation updates as user preferences evolve
- Hybrid approach combining collaborative and content-based filtering
- Cross-community discovery to explore adjacent taste preferences
- Community analytics and visualization dashboard
- Cold-start solutions for new users with minimal ratings

---

### Becuase "Birds of a Feather" Stick Together

_“Just like Billie sings, birds of a feather, we should stick together — and here, users who rate movies similarly naturally cluster together.” 🎵_

![bird of a feather](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbW5mdnVuYTF4czltNHg1anVlamlmc2RkMHlkNHZwOWtzN2NiN25lNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/btjXY59RCYum69L64R/giphy.gif)
