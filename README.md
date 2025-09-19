# Recommendation System using PySpark's ALS Algorithm

Personalized movie recommendations powered by **PySpark’s ALS (Alternating Least Squares)** collaborative filtering.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![ALS Algorithm](https://img.shields.io/badge/ML-ALS%20Collaborative%20Filtering-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 1. Story Behind the Project

### Because _“Birds of a Feather”_ Stick Together

_“Just like Billie sings, birds of a feather, we should stick together — and here, users who rate movies similarly naturally cluster together.” 🎵_

![Birds of a feather](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbW5mdnVuYTF4czltNHg1anVlamlmc2RkMHlkNHZwOWtzN2NiN25lNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/btjXY59RCYum69L64R/giphy.gif)

Movies are more than ratings — they reveal patterns of taste. ALS uncovers these hidden clusters, grouping users with similar preferences to generate meaningful recommendations.


---

## 2. Project Overview 

This system applies **collaborative filtering** to IMDb ratings data, using ALS to identify hidden “taste communities.” Users receive personalized movie recommendations through a Streamlit interface.

**Key capabilities:**

- Efficiently process **11M+ titles** and millions of ratings
- Discover **latent user communities** with ALS
- Deliver **real-time personalized recommendations**



---

## 3. Architecture

```
📦 Movie Recommender
├── 🗃️  utils/DataLoader           # Load & clean IMDb datasets
├── 👤  utils/User                 # Manage user profiles & ratings
├── 🤖  utils/Recommender          # ALS collaborative filtering & predictions
├── 🔄  utils/IMDbRatingsConverter # Generate synthetic ratings
└── 🌐  app.py                     # Streamlit web interface
```

---

## 4. Tech Stack

- **PySpark** → Distributed ML model training
- **ALS Algorithm** → Matrix factorization for recommendations
- **PostgreSQL** → Store IMDb data, user profiles, ratings
- **Streamlit** → Web interface for ratings & recommendations
- **Pandas** → Data analysis and community insights

---

## 5. Quick Start

### Prerequisites

- Python 3.9+
- PySpark 3.5.0+
- PostgreSQL 14+
- Java 8+ (required by PySpark)

### Installation

```bash
git clone git@github.com:jgchoti/recommender-systems-pyspark.git
cd recommender-systems-pyspark
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Environment setup

Create a `.env` file:

```env
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password
DB_HOST=localhost
```

### Database

```sql
CREATE DATABASE imdb_recommendation;
```

### Run the app

```bash
# Load IMDb data into PostgreSQL (first time only)
python utils/data_loader.py

# Generate synthetic user ratings from IMDb aggregates (first time only)
python utils/IMDbRatingsConverter.py

# Launch Streamlit app
streamlit run app.py
```

App runs at: `http://localhost:8501`

---

## 6. How It Works

### ALS Algorithm

- **Latent factor discovery**: Finds hidden taste dimensions
- **Community clustering**: Groups users with similar rating patterns
- **Recommendation**: Suggests titles highly rated by users in the same cluster

### Training Data

IMDb aggregated ratings are mapped to synthetic user profiles:

- 8.0+ → Users rating 4–5 stars
- 7.0–7.9 → Users rating 3–4 stars
- 6.0–6.9 → Users rating 2–4 stars

### Pipeline

1. User rates movies in Streamlit
2. ALS maps them to a taste community
3. System retrieves titles liked by similar users
4. Recommendations delivered with confidence scores

---

## 7. Application Features

**System Management**

- Load IMDb data into PostgreSQL
- Generate sample users & ratings

**User Management**

- Create accounts, login/logout
- Manage profiles and history

**Movie Rating**

- Browse/filter IMDb titles
- Rate 1–5 stars, stored instantly

**Recommendation Engine**

- Train ALS model
- Display personalized recommendations with predicted ratings

**Profile & Statistics**

- Track rating history
- Monitor recommendation performance

---

## 8. Learning Highlights

- **Latent factors capture hidden taste dimensions**: Users with similar rating patterns align along the same factors in the matrix factorization space.

- **ALS learns features implicitly**: It infers user and item embeddings without requiring manual feature engineering.

- **Scales efficiently**: The algorithm handles millions of users and items through distributed matrix factorization in Spark.

---

## 9. Future Improvements

- Real-time model updates as users add ratings
- Hybrid model: collaborative + content-based filtering
- Visualization dashboard for community insights
- Cold-start strategies for new users
