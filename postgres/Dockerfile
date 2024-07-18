FROM postgres:latest

# Install required packages and dependencies
RUN apt-get update && \
    apt-get install -y \
    postgresql-server-dev-all \
    build-essential \
    git \
    libxml2-dev \
    libxslt-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libpq-dev \
    libprotobuf-c-dev \
    protobuf-c-compiler \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Clone, build, and install pgvector
RUN git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && make install && \
    cd .. && rm -rf pgvector

# Install pg_trgm extension
RUN apt-get update && \
    apt-get install -y postgresql-contrib && \
    rm -rf /var/lib/apt/lists/*

# Create the extensions in the database
COPY init-db.sh /docker-entrypoint-initdb.d/

# Start the postgres server
CMD ["postgres"]
