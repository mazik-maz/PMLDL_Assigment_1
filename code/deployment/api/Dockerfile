# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /api

# Copy the current directory contents into the container
COPY . /api

# Install FastAPI and other dependencies
RUN pip install fastapi uvicorn joblib scikit-learn pydantic numpy

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
