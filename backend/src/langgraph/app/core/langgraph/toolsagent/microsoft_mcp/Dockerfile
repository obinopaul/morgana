# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, a fast Python package installer
RUN pip install uv

# Copy the project definition file first to leverage Docker layer caching
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system .

# Copy the rest of your application's source code
COPY ./src ./src
COPY ./authenticate.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the MCP server using its defined entry point from pyproject.toml
# This command is equivalent to running 'microsoft-mcp'
CMD ["uv", "run", "microsoft-mcp"]

