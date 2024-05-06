# SimilityVectorEmbedding

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### Overview
This project demonstrates the power of natural language processing combined with vector databases to efficiently find similar movies based on their descriptions and metadata. Using technologies such as PostgreSQL with pgvector and advanced NLP models, this project provides a robust solution for similarity searches in large datasets.

### Installation
Install all required libraries and dependencies:
```bash
pip install transformers psycopg2 numpy boto3 torch scikit-learn matplotlib nltk sentence-transformers
```

### Usage
Run the main Python script or use the Jupyter notebook included in the repository to interact with the project's functionalities. This could include generating embeddings, inserting data into the database, or querying for similar movies.

### Database Setup and Data Handling

#### Starting PostgreSQL with pgvector
Ensure PostgreSQL is running with the pgvector extension, which allows efficient handling of vector data for fast similarity searching:
```bash
docker-compose up -d
```
### Database

![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/86c8b625-8fc6-4727-9c9c-efaf85cc88d1)


#### Insert Data
Use the `insert_movies(movie_data, embeddings)` function to load movie data along with generated embeddings into the database.

### Working with Embeddings
Discuss how embeddings are generated using models like BERT or Sentence Transformers, and how they are utilized within pgvector to perform fast and efficient cosine similarity searches.

### Restoring a Database Backup
Provide steps to restore a PostgreSQL database from a backup file using Docker:
```bash
# Stop the current PostgreSQL container
docker-compose down

# Start a new PostgreSQL instance
docker-compose up -d

# Restore the database from a backup file
docker exec -i [container_name] pg_restore -U [username] -d [database_name] < [backup_file_path]
```

### Finding Similar Movies
Detail the SQL queries and Python functions used to find movies similar to a given query movie based on embeddings similarity.

### Understanding Vector Querying and Cosine Similarity

#### Vector Querying with pgvector
Pgvector is a PostgreSQL extension that facilitates efficient storage and querying of high-dimensional vectors. In this project, we leverage pgvector to handle vector data derived from movie embeddings. These embeddings represent the semantic content of movie descriptions and metadata, allowing for advanced querying capabilities like nearest neighbor searches.

#### Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors. This metric is widely used in natural language processing to assess how similar two documents (or in this case, movie descriptions) are irrespective of their size. Mathematically, it's defined as:

\[ \text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} \]

where \(\mathbf{A}\) and \(\mathbf{B}\) are two vectors, and \(\|\mathbf{A}\|\) and \(\|\mathbf{B}\|\) are their norms.

![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/c86756b1-0afe-4f52-b547-00cf4ee81aab)


#### Implementing Cosine Similarity in PostgreSQL with pgvector
Pgvector supports several distance metrics, including cosine similarity (denoted as `<=>` in SQL). By utilizing this function, we can perform fast cosine distance calculations directly within SQL queries, which is critical for efficient similarity searches. Here’s how you can find similar movies based on cosine similarity:

```sql
SELECT title, embedding
FROM movies
ORDER BY embedding <=> (SELECT embedding FROM movies WHERE title = %s) ASC
LIMIT 10;
```

This SQL command retrieves the ten most similar movies to a given movie based on their embeddings' cosine similarity.

#### Other Distance Functions Supported by pgvector
Pgvector also supports other distance metrics such as L2 (Euclidean), L1 (Manhattan), and Dot Product. Each of these metrics can be selected based on the specific needs of your query or the characteristics of your data. Here’s how you might use these metrics:

- **L2 Distance (Euclidean)**: Suitable for measuring the absolute differences between vectors.
- **L1 Distance (Manhattan)**: Useful in high-dimensional data spaces.

#### JSON

![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/608bc866-b092-4e73-81d8-be37ca5ad800)

#### Movie Entry
Here is an example of how a movie is represented in the `movies.json`:
```json
{
  "titre": "George of the Jungle",
  "annee": "1997",
  "pays": "USA",
  "langue": "English",
  "duree": "92",
  "resume": "George grows up in the jungle raised by apes. Based on the Cartoon series.",
  "genre": ["Action", "Adventure", "Comedy", "Family", "Romance"],
  "realisateur": {"_id": "918873", "__text": "Sam Weisman"},
  "scenariste": ["Jay Ward", "Dana Olsen"],
  "role": [
    {"acteur": {"_id": "409", "__text": "Brendan Fraser"}, "personnage": "George of the Jungle"},
    {"acteur": {"_id": "5182", "__text": "Leslie Mann"}, "personnage": "Ursula Stanhope"}
  ],
  "poster": "https://m.media-amazon.com/images/M/MV5BNTdiM2VjYjYtZjEwNS00ZWU5LWFkZGYtZGYxMDcwMzY1OTEzL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTczNjQwOTY@._V1_SY150_CR0,0,101,150_.jpg",
  "_id": "119190"
}
```

## Reference
- https://www.youtube.com/watch?v=QdDoFfkVkcw
- https://www.machinelearningplus.com/nlp/cosine-similarity/
- https://www.youtube.com/watch?v=Yhtjd7yGGGA
- https://sbert.net
- https://huggingface.co/spaces/mteb/leaderboard
- https://github.com/rabbit-hole-syndrome/open-source-embeddings
- https://sbert.net/docs/pretrained_models.html
- https://cookbook.openai.com/examples/visualizing_embeddings_in_2d
- https://platform.openai.com/docs/guides/embeddings
