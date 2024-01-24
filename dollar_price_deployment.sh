# @ Author: Ahmad Shayaan
# @ Create Time: 2023-07-28 17:17:38
# @ Modified by: Mitas Ray
# @ Modified time: 2023-01-23

#!/bin/sh
who
# Changing directory and training the model
echo "Training model"
/opt/conda/bin/python /home/mitas/ficc_python/automated_training_dollar_price_model.py
if [ $? -ne 0 ]; then
  echo "Python script failed with exit code $?"
  exit 1
fi
echo "Model trained"

#Getting the endpoint ID we want to deploy the model on
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-east4 --format='value(ENDPOINT_ID)' --filter=display_name='dollar_price_model')
# ENDPOINT_ID=$(gcloud ai endpoints list --region=us-east4 --format='value(ENDPOINT_ID)' --filter=display_name='test')

#Unzip model and uploading it to automated training bucket
TIMESTAMP=$(date +%m-%d)
MODEL_NAME='dollar-model'-${TIMESTAMP}
echo "Unzipping model $MODEL_NAME"
gsutil cp -r gs://ahmad_data/model_dollar_price.zip /home/mitas/trained_models/dollar_price_models/model_dollar_price.zip
unzip /home/mitas/trained_models/dollar_price_models/model_dollar_price.zip -d /home/mitas/trained_models/dollar_price_models/$MODEL_NAME
if [ $? -ne 0 ]; then
  echo "Unzipping failed with exit code $?"
  exit 1
fi

echo "Uploading model to bucket"
gsutil cp -r /home/mitas/trained_models/dollar_price_models/$MODEL_NAME gs://automated_training
if [ $? -ne 0 ]; then
  echo "Uploading model failed with exit code $?"
  exit 1
fi


echo $ENDPOINT_ID
echo $MODEL_NAME
echo "Uploading model to vertex ai"
gcloud beta ai models upload --region=us-east4 --display-name=$MODEL_NAME --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8:latest --artifact-uri=gs://automated_training/$MODEL_NAME
if [ $? -ne 0 ]; then
  echo "Failed to deploy model on vertex ai exited with code $?"
  exit 1
fi

NEW_MODEL_ID=$(gcloud ai models list --region=us-east4 --format='value(name)' --filter='displayName'=$MODEL_NAME)
echo $NEW_MODEL_ID
echo $MODEL_NAME
echo "Deploying to endpoint"
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=us-east4 --display-name=$MODEL_NAME --model=$NEW_MODEL_ID --machine-type=n1-standard-2  --accelerator=type=nvidia-tesla-p4,count=1 --min-replica-count=1 --max-replica-count=1

sudo shutdown -h now
