# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /soccer-mcp-server

# Copy the current directory contents into the container
COPY . /soccer-mcp-server

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your app will run on
EXPOSE 8000

# Run the server when the container launches
CMD ["python", "soccer_server.py"]

# docker build -t nba_server .
# docker run -p 4000:5000 nba_server

