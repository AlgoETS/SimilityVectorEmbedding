# Simility Vector Embedding

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### Overview
This project demonstrates the power of natural language processing combined with vector databases to efficiently find similar movies based on their descriptions and metadata. Using technologies such as PostgreSQL with pgvector and advanced NLP models, this project provides a robust solution for similarity searches in large datasets.


![Animation8](https://github.com/user-attachments/assets/2bd8ae0c-1b24-4774-865d-244b58a4f362)
![Animation13](https://github.com/user-attachments/assets/1f9a2fa3-b195-4f2d-bb38-eb4a218e2eaf)
![Animation9](https://github.com/user-attachments/assets/e688a0dd-61e3-493e-aee1-b582f313c182)

![Capture3](https://github.com/user-attachments/assets/ce23c1e3-74be-4fd8-9333-1e5c7b9cc8bf)


### Usage
Use the Jupyter notebook. This could include generating embeddings, inserting data into the database, or querying for similar movies.

### Database Setup and Data Handling
### Database

![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/86c8b625-8fc6-4727-9c9c-efaf85cc88d1)


### Working with Embeddings
Discuss how embeddings are generated using models like BERT or Sentence Transformers, and how they are utilized within pgvector to perform fast and efficient cosine similarity searches.

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

## IMDB databased

https://developer.imdb.com/non-commercial-datasets/

![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/ec763cbf-2975-444f-b065-b9be7372a6dd)


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
- https://colab.research.google.com/github/qdrant/examples/blob/master/qdrant_101_audio_data/03_qdrant_101_audio.ipynb
- https://qdrant.tech/documentation/examples/recommendation-system-ovhcloud/
- https://colab.research.google.com/github/qdrant/examples/blob/master/qdrant_101_text_data/qdrant_and_text_data.ipynb
- https://www.youtube.com/watch?v=Vkazja71BkA
- https://www.youtube.com/watch?v=p1LtVo_1Q7A
- https://espace.etsmtl.ca/id/eprint/2576/2/BOUCHER_CHARBONNEAU_Kristof.pdf
