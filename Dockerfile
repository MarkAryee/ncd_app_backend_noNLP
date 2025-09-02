# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR .

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your app files
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
