# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install necessary packages including Java from the adoptopenjdk repository
RUN apt-get update && \
    apt-get install -y wget gnupg && \
    mkdir -p /etc/apt/keyrings && \
    wget -qO /etc/apt/keyrings/adoptopenjdk.asc https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public && \
    echo "deb [signed-by=/etc/apt/keyrings/adoptopenjdk.asc] https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ buster main" | tee /etc/apt/sources.list.d/adoptopenjdk.list && \
    apt-get update && \
    apt-get install -y adoptopenjdk-11-hotspot && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/adoptopenjdk-11-hotspot-amd64

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container at /app
ADD indexing.py /app/indexing.py
ADD retrieve.py /app/retrieve.py
ADD re-rank.py /app/re-rank.py
ADD requirements.txt /app/requirements.txt
ADD run.sh /app/run.sh

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Make the run.sh script executable
RUN chmod +x /app/run.sh

# Execute the run.sh script
ENTRYPOINT [ "/app/run.sh" ]

