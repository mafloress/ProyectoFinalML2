# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script, model, and data into the container at /app
COPY scripts/batch_inference_script.py /app/
COPY models/trained_model.pkl /app/
COPY data/processed/processed_lending_club_data.csv /app/
# Note: Bundling data like this is okay for an exercise. 
# In production, data would likely be fetched from a DB/S3 or mounted.

# Define environment variables (if any, not strictly needed for this simple script)
# ENV NAME World

# Run batch_inference_script.py when the container launches
CMD ["python", "/app/batch_inference_script.py"]
