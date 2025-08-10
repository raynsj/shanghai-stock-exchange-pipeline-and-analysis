# Use an official Python image as a starting point
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy your notebook and any other files into the container
COPY ./notebooks /app/notebooks

# Expose the port Jupyter runs on
EXPOSE 8888

# The command to run when the container starts
# This starts the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks"]