# @ Author: Ahmad Shayaan
# @ Create date: 2023-07-28
# @ Modified by: Mitas Ray
# @ Modified date: 2023-04-10
echo "If there are errors, visit: https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8"
echo "Search for warnings in the logs (even on a successful training procedure) and investigate"

#!/bin/sh
who
HOME='/home/mitas'
TRAINED_MODELS_PATH="$HOME/trained_models/dollar_price_models"
# Create dates before training so that in case the training takes too long and goes into the next day, the date is correct
DATE_WITH_YEAR=$(date +%Y-%m-%d)
DATE_WITHOUT_YEAR=$(date +%m-%d)
TRAINING_LOG_PATH="$HOME/training_logs/dollar_price_training_$DATE_WITH_YEAR.log"
MODEL="dollar_price"

# Activate the virtual environment for Python3.10 (/usr/local/bin/python3.10) that contains all of the packages; to see all versions of Python use command `whereis python`
# If venv_py310 does not exist in `ficc_python/`, then in `ficc_python/` run `/usr/local/python3.10 -m venv venv_py310` and `source venv_py310/bin/activate` followed by `pip install -r requirements_py310.txt`
source $HOME/ficc_python/venv_py310/bin/activate
python --version

# Training the model
python $HOME/ficc_python/automated_training_dollar_price_model.py
if [ $? -ne 0 ]; then
  echo "automated_training_dollar_price_model.py script failed with exit code $?"
  python $HOME/ficc_python/clean_training_log.py $TRAINING_LOG_PATH
  python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model training failed. See attached logs for more details. However, if there is not enough new trades on the previous business day, then this is the desired behavior."
  exit 1
fi
echo "Model trained"

# Cleaning the logs to make more readable
python $HOME/ficc_python/clean_training_log.py $TRAINING_LOG_PATH

# Unzip model and uploading it to automated training bucket
MODEL_NAME='dollar-model'-${DATE_WITHOUT_YEAR}
echo "Unzipping model $MODEL_NAME"
gsutil cp -r gs://automated_training/model_dollar_price.zip $TRAINED_MODELS_PATH/model_dollar_price.zip
unzip $TRAINED_MODELS_PATH/model_dollar_price.zip -d $TRAINED_MODELS_PATH/$MODEL_NAME
if [ $? -ne 0 ]; then
  echo "Unzipping failed with exit code $?"
  python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Unzipping model failed. See attached logs for more details."
  exit 1
fi

echo "Uploading model to bucket"
gsutil cp -r $TRAINED_MODELS_PATH/$MODEL_NAME gs://automated_training
if [ $? -ne 0 ]; then
  echo "Uploading model to bucket failed with exit code $?"
  python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Uploading model to bucket failed. See attached logs for more details."
  exit 1
fi

# Getting the endpoint ID we want to deploy the model on
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-east4 --format='value(ENDPOINT_ID)' --filter=display_name='dollar_price_model')

echo "ENDPOINT_ID $ENDPOINT_ID"
echo "MODEL_NAME $MODEL_NAME"
echo "Uploading model to Vertex AI"
gcloud beta ai models upload --region=us-east4 --display-name=$MODEL_NAME --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest --artifact-uri=gs://automated_training/$MODEL_NAME
if [ $? -ne 0 ]; then
  echo "Model upload to Vertex AI failed with exit code $?"
  python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model upload to Vertex AI failed. See attached logs for more details."
  exit 1
fi

NEW_MODEL_ID=$(gcloud ai models list --region=us-east4 --format='value(name)' --filter='displayName'=$MODEL_NAME)
echo "NEW_MODEL_ID $NEW_MODEL_ID"
echo "Deploying to endpoint"
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=us-east4 --display-name=$MODEL_NAME --model=$NEW_MODEL_ID --machine-type=n1-standard-2 --accelerator=type=nvidia-tesla-p4,count=1 --min-replica-count=1 --max-replica-count=1
if [ $? -ne 0 ]; then
  echo "Model deployment to Vertex AI failed with exit code $?"
  python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model deployment to Vertex AI failed. See attached logs for more details."
  exit 1
fi

# removing temporary files
rm $TRAINED_MODELS_PATH/model_dollar_price.zip
gsutil rm -r gs://automated_training/model_dollar_price.zip

python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "No detected errors. Logs attached for reference."

# Deactivate the virtual environment
deactivate

sudo shutdown -h now
