# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt  


# Copy the rest of the application code
COPY . .

# Expose  the port your Flask app will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]
