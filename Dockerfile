# Multi-stage build.  To build the container, you will need to supply your
# OPENAI_API_KEY to load the vector database within the final container
#   docker build --build-arg OPENAI_API_KEY=$OPENAI_API_KEY -f Dockerfile -t agr .

# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Ingestion script needs OpenAI embeddings to insert into vector database.
# Pull OPENAI_API_KEY from arguments
ARG OPENAI_API_KEY

# Set environment variable for ingestion script
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run ingestion script to populate vector database
RUN python ingestion.py

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Copy application over
COPY --from=builder /app /app

# Copy package libraries over
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Set working directory
WORKDIR /app

# Launch uvicorn
CMD python -m uvicorn llm:app --reload --port=8000 --host=0.0.0.0
