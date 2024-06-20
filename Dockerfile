# Use an official Python runtime as a parent image
# docker build --build-arg OPENAI_API_KEY=$OPENAI_API_KEY -f Dockerfile -t agr .
FROM python:3.10-slim

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
RUN pip install -r requirements.txt

# Run ingestion script to populate vector database
RUN python ingestion.py

# Start endpoint on port 8000
CMD uvicorn llm:app --reload --port=8000 --host=0.0.0.0

