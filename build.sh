git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
docker build --build-arg PG_MAJOR=16 -t builder/pgvector .
cd ..
docker-compose up -d
