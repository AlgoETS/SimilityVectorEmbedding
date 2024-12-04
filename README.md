# Similarity Vector Embedding

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The **Similarity Vector Embedding** project utilizes Natural Language Processing (NLP) and vector databases to efficiently identify and recommend similar movies based on their descriptions and metadata. By leveraging PostgreSQL with the pgvector extension and advanced NLP models like BERT and Sentence Transformers, this project offers a scalable solution for performing similarity searches within large movie datasets. This system is ideal for enhancing recommendation engines, improving content discovery, and organizing extensive media collections.


![Workflow Animation 1](https://github.com/user-attachments/assets/2bd8ae0c-1b24-4774-865d-244b58a4f362)

*Figure 1: Gradio App example.*

![Workflow Animation 2](https://github.com/user-attachments/assets/1f9a2fa3-b195-4f2d-bb38-eb4a218e2eaf)

*Figure 2: Embedding generation process using Sentence Transformers.*

![Workflow Animation 3](https://github.com/user-attachments/assets/e688a0dd-61e3-493e-aee1-b582f313c182)

*Figure 3: Similarity search and recommendation pipeline Qdrant.*


#### Implementing Cosine Similarity in PostgreSQL with pgvector
Pgvector supports several distance metrics, including cosine similarity (denoted as <=> in SQL). By utilizing this function, we can perform fast cosine distance calculations directly within SQL queries, which is critical for efficient similarity searches. Here’s how you can find similar movies based on cosine similarity:

## Getting Started

### Prerequisites

- **Python 3.8**
- **PostgreSQL**
- **pgvector Extension**
- **Jupyter Notebook**

### Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/AlgoETS/SimilityVectorEmbedding.git
    cd SimilityVectorEmbedding
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up PostgreSQL with pgvector:**
    - **Install PostgreSQL:** [Download here](https://www.postgresql.org/download/)
    - **Install pgvector Extension:**
        ```bash
        sudo apt install postgresql-14-pgvector
        ```
        *Or build from source:*
        ```bash
        git clone https://github.com/pgvector/pgvector.git
        cd pgvector
        make
        sudo make install
        ```

4. **Create Database and Enable pgvector:**
    ```sql
    CREATE DATABASE movies_db;
    \c movies_db
    CREATE EXTENSION vector;
    ```

5. **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `Similarity_Vector_Embedding.ipynb` and follow the instructions to generate embeddings, insert data, and perform similarity queries.

## Usage

![Architecture Diagram](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/86c8b625-8fc6-4727-9c9c-efaf85cc88d1)

*Figure 4: System architecture integrating PostgreSQL, pgvector, and NLP models.*

![Similarity Search Example](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/ec763cbf-2975-444f-b065-b9be7372a6dd)

*Figure 5: Example of cosine similarity results for the movie "Inception".*


![image](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/c86756b1-0afe-4f52-b547-00cf4ee81aab)

#### Implementing Cosine Similarity in PostgreSQL with pgvector
Pgvector supports several distance metrics, including cosine similarity (denoted as <=> in SQL). By utilizing this function, we can perform fast cosine distance calculations directly within SQL queries, which is critical for efficient similarity searches. Here’s how you can find similar movies based on cosine similarity:

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

## Database Schema

```sql
CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    year INT,
    country VARCHAR(100),
    language VARCHAR(100),
    duration INT,
    summary TEXT,
    genres TEXT[],
    director JSONB,
    screenwriters TEXT[],
    roles JSONB,
    poster_url TEXT,
    embedding VECTOR(768) -- Adjust dimension based on NLP model
);
```

## Data Example

```json
{
  "title": "Inception",
  "year": "2010",
  "country": "USA",
  "language": "English",
  "duration": "148",
  "summary": "A skilled thief is given a chance at redemption if he can successfully perform an inception.",
  "genres": ["Action", "Sci-Fi", "Thriller"],
  "director": {"_id": "123456", "__text": "Christopher Nolan"},
  "screenwriters": ["Christopher Nolan"],
  "roles": [
    {"actor": {"_id": "78910", "__text": "Leonardo DiCaprio"}, "character": "Cobb"},
    {"actor": {"_id": "111213", "__text": "Joseph Gordon-Levitt"}, "character": "Arthur"}
  ],
  "poster_url": "https://m.media-amazon.com/images/I/51G8J1XnFQL._AC_SY445_.jpg",
  "id": "54321"
}
```


### Generating Embeddings

Use Sentence Transformers to generate embeddings for movie descriptions:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return model.encode(text).tolist()
```

### Inserting Data into the Database

Populate the `movies` table with movie data and their embeddings:

```python
import json
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="movies_db",
    user="your_username",
    password="your_password",
    host="localhost"
)
cursor = conn.cursor()

# Load movie data
with open('movies.json', 'r') as file:
    movies = json.load(file)

# Insert movies into the database
for movie in movies:
    cursor.execute("""
        INSERT INTO movies (title, year, country, language, duration, summary, genres, director, screenwriters, roles, poster_url, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        movie['titre'],
        movie['annee'],
        movie['pays'],
        movie['langue'],
        movie['duree'],
        movie['resume'],
        movie['genre'],
        json.dumps(movie['realisateur']),
        json.dumps(movie['scenariste']),
        json.dumps(movie['role']),
        movie['poster'],
        generate_embedding(movie['resume'])
    ))

conn.commit()
cursor.close()
conn.close()
```

### Finding Similar Movies

Retrieve movies similar to a given title using cosine similarity:

```python
import psycopg2

def find_similar_movies(movie_title, top_k=10):
    conn = psycopg2.connect(
        dbname="movies_db",
        user="your_username",
        password="your_password",
        host="localhost"
    )
    cursor = conn.cursor()
    query = """
    SELECT title
    FROM movies
    WHERE title != %s
    ORDER BY embedding <=> (
        SELECT embedding FROM movies WHERE title = %s
    ) ASC
    LIMIT %s;
    """
    cursor.execute(query, (movie_title, movie_title, top_k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [movie[0] for movie in results]

# Example usage
similar_movies = find_similar_movies("Inception")
print(similar_movies)
```

## IMDB databased

https://developer.imdb.com/non-commercial-datasets/
![System Architecture](https://github.com/AlgoETS/SimilityVectorEmbedding/assets/13888068/86c8b625-8fc6-4727-9c9c-efaf85cc88d1)
*Figure 6: IMDB .*****


## Language Models Used

| Model Name                           | Description                                                         | Source                                                                                      |
|--------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **BERT**                             | Bidirectional Encoder Representations from Transformers.           | [BERT on Hugging Face](https://huggingface.co/bert-base-uncased)                            |
| **Sentence Transformers**            | Models optimized for generating sentence-level embeddings.          | [Sentence Transformers](https://sbert.net/)                                                |
| **all-MiniLM-L6-v2**                 | A lightweight and efficient Sentence Transformer model.             | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)          |
| **RoBERTa**                          | A robustly optimized BERT pretraining approach.                    | [RoBERTa on Hugging Face](https://huggingface.co/roberta-base)                             |
| **DistilBERT**                       | A distilled version of BERT, smaller and faster while retaining performance. | [DistilBERT on Hugging Face](https://huggingface.co/distilbert-base-uncased)                |
| **XLNet**                            | Generalized autoregressive pretraining for language understanding. | [XLNet on Hugging Face](https://huggingface.co/xlnet-base-cased)                           |
| **T5**                               | Text-to-Text Transfer Transformer for various NLP tasks.           | [T5 on Hugging Face](https://huggingface.co/t5-base)                                       |
| **Electra**                          | Efficient pretraining approach replacing masked tokens with generators. | [Electra on Hugging Face](https://huggingface.co/google/electra-base-discriminator)         |
| **Longformer**                       | Transformer model optimized for long documents.                     | [Longformer on Hugging Face](https://huggingface.co/allenai/longformer-base-4096)           |
| **MiniLM-L12-v2**                    | A compact and efficient model for sentence embeddings.              | [MiniLM-L12-v2 on Hugging Face](https://huggingface.co/sentence-transformers/minilm-l12-v2) |
| **SBERT DistilRoBERTa**              | A distilled version of RoBERTa for efficient sentence embeddings.  | [SBERT DistilRoBERTa on Hugging Face](https://huggingface.co/sentence-transformers/distilroberta-base) |
| **MPNet**                            | Masked and Permuted Pre-training for Language Understanding.        | [MPNet on Hugging Face](https://huggingface.co/microsoft/mpnet-base)                       |
| **ERNIE**                            | Enhanced Representation through Knowledge Integration.              | [ERNIE on Hugging Face](https://huggingface.co/nghuyong/ernie-2.0-en)                        |
| **DeBERTa**                          | Decoding-enhanced BERT with disentangled attention.                 | [DeBERTa on Hugging Face](https://huggingface.co/microsoft/deberta-base)                    |
| **SBERT paraphrase-MiniLM-L6-v2**    | A Sentence Transformer model fine-tuned for paraphrase identification. | [paraphrase-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) |


**Personal Preference:**

I personally prefer using **T5-small** and the **MiniLM** series models due to their excellent balance between performance and computational efficiency.


## References

### Tutorials and Guides
- [Cosine Similarity in NLP](https://www.machinelearningplus.com/nlp/cosine-similarity/)
- [Visualizing Embeddings in 2D](https://cookbook.openai.com/examples/visualizing_embeddings_in_2d)
- [Qdrant Text Data Example](https://colab.research.google.com/github/qdrant/examples/blob/master/qdrant_101_text_data/qdrant_and_text_data.ipynb)
- [Recommendation System with Qdrant](https://qdrant.tech/documentation/examples/recommendation-system-ovhcloud/)

### Documentation
- [pgvector GitHub Repository](https://github.com/pgvector/pgvector)
- [Sentence Transformers Documentation](https://sbert.net/docs/pretrained_models.html)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### Videos
- [Understanding Cosine Similarity](https://www.youtube.com/watch?v=QdDoFfkVkcw)
- [Implementing Vector Databases](https://www.youtube.com/watch?v=Yhtjd7yGGGA)
- [Embedding Models Explained](https://www.youtube.com/watch?v=Vkazja71BkA)
- [Advanced Embedding Techniques](https://www.youtube.com/watch?v=p1LtVo_1Q7A)

### Additional Resources
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Open Source Embeddings Collection](https://github.com/rabbit-hole-syndrome/open-source-embeddings)
- [Qdrant Audio Data Example](https://colab.research.google.com/github/qdrant/examples/blob/master/qdrant_101_audio_data/03_qdrant_101_audio.ipynb)
- [Research Paper on Embeddings](https://espace.etsmtl.ca/id/eprint/2576/2/BOUCHER_CHARBONNEAU_Kristof.pdf)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please open an issue on the GitHub repository or contact [antoine@antoineboucher.info](mailto:antoine@antoineboucher.info)
