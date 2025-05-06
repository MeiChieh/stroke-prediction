# base image
FROM python:3.10-slim

# set the working directory in the container
WORKDIR /stroke_prediction_app

# copy the requirements.txt file into the container at /stroke_prediction_app
COPY requirements.txt /stroke_prediction_app

# copy best model to app
COPY best_model.pkl /stroke_prediction_app/

# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# install uvicorn
RUN pip install uvicorn

# copy the FastAPI code into the container at /stroke_prediction_app
COPY fast_api_deploy.py /stroke_prediction_app/

# command to run the FastAPI application
CMD ["uvicorn", "fast_api_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
