# Salary Classification

This repository contains a simple salary classification on us census data. It was used during a udacity training to practice:
* dvc
* pytest
* render 
* github actions
* fastapi
* general ml material like a model card

In the folder `starter/screenshots`a couple of images show the realized steps.

Install requirements within a conda env with `cd starter && pip install -r requirements.txt`.
You can run training with `cd starter/starter && python train_model.py`. 
You can run tests with `cd starter/starter && PYTHONPATH=$(pwd) pytest tests`. 
You can start the fastapi server with `cd starter/starter && uvicorn main:app --host 0.0.0.0 --port 8000`.