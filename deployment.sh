#!/bin/sh

#Changing directory adn training the model
echo "Training model"
/opt/conda/bin/python /home/ahmad/ficc_python/automated_training.py > /home/ahmad/output.txt
echo "Model trained"

# Getting the endpoint ID we want to deploy the model on
# ENDPOINT_ID=$(gcloud ai endpoints list --region=us-east4 --format='value(ENDPOINT_ID)' --filter=display_name='new_attention_model')

# # Unzipt model and uploading it to automated training bucket
# TIMESTAMP=$(date +%m-%d)
# MODEL_NAME='model'-${TIMESTAMP}
# echo "Unziping model $MODEL_NAME"
# gsutil cp -r gs://ahmad_data/model.zip /home/ahmad/ficc_python/model.zip
# unzip /home/ahmad/ficc_python/model.zip -d /home/ahmad/trained_models/$MODEL_NAME

# echo "Uploading model to bucket"
# gsutil cp -r /home/ahmad/trained_models/$MODEL_NAME gs://automated_training

# echo $ENDPOINT_ID
# echo $MODEL_NAME
# echo "Deploying model"
# echo "gcloud beta ai models upload --region=us-east4 --display-name=$MODEL_NAME --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest --artifact-uri=gs://automated_training/$MODEL_NAME"
# echo "Deploying endpoint"
# gcloud ai endpoints deploy-model $ENDPOINT_ID --region=us-east4 --display-name=$NAME --machine-type=n1-standard-2 --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100