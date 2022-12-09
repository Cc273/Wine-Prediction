FROM python:3.8-slim-buster

# Install dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Install Java
RUN apt-get update && apt-get install -y openjdk-11-jdk && rm -rf /var/lib/apt/lists/*

# Install pyspark and numpy
RUN pip3 install pyspark numpy

# Copy the current directory contents into the container at /app
COPY ./predict.py /app/predict.py

# Set the working directory to /app
WORKDIR /app

# Run predict.py when the container launches
CMD ["python3", "predict.py"]

