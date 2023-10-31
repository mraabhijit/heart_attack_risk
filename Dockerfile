# Select the base image
FROM python:3.11.5-slim 

# Install Pipenv for environment management
RUN pip install pipenv

# Create new directory "app" and change directory
WORKDIR /app
# Copy files to './' which means to current directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# Create virtual environment in local system using --system --deploy
RUN pipenv install --system --deploy

# Copy predict.py and model.bin files to /app
COPY ["predict.py", "model.bin", "./"]

# Expose the port for communication
EXPOSE 8080

# Specify entrypoint for the gunicorn command to work with
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict:app"]