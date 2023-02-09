#!/bin/sh

#unzip /home/ahmad/ficc_python/model.zip

NAME='test'
ENDPOINT_NAME=$(gcloud ai endpoints list --region=us-east4 --format='value(DISPLAY_NAME)')
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-east4 --format='value(ENDPOINT_ID)' --filter=display_name=${NAME})

TIMESTAMP=$(date +%m-%d)
MODEL_NAME='model'-${TIMESTAMP}


echo $ENDPOINT_NAME
echo $ENDPOINT_ID
echo $MODEL_NAME
echo "Deploying model"
gcloud beta ai models upload --region=us-east4 --display-name=$MODEL_NAME --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest --artifact-uri=gs://automated_training/model
echo "Deploying endpoint"
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=us-east4 --display-name=$NAME --machine-type=n1-standard-2 --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100