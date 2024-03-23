# @ Author: Ahmad Shayaan
# @ Create date: 2023-07-28
# @ Modified by: Mitas Ray
# @ Modified date: 2023-03-22
echo "If there are errors, visit: https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8"
echo "Search for warnings in the logs (even on a successful training procedure) and investigate"

#!/bin/sh
who
HOME='/home/mitas'
TRAINED_MODELS_PATH="$HOME/trained_models/yield_spread_models"

DATE_STRINGS="2024-01-26 2024-02-02"

for DATE_STRING in $DATE_STRINGS; do
  DATE_WITH_YEAR=$(date -d "$DATE_STRING" +%Y-%m-%d)
  DATE_WITHOUT_YEAR=$(date -d "$DATE_STRING" +%m-%d)
  TRAINING_LOG_PATH="$HOME/training_logs/retrain-yield_spread_training.log"
  MODEL="yield_spread"

  # Changing directory and training the model
  /opt/conda/bin/python $HOME/ficc_python/automated_training_yield_spread_model.py $DATE_WITH_YEAR
  if [ $? -ne 0 ]; then
    echo "automated_training_yield_spread_model.py script failed with exit code $?"
    /opt/conda/bin/python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model training failed. See attached logs for more details."
    exit 1
  fi
  echo "Model trained"

  # Cleaning the logs to make more readable
  /opt/conda/bin/python $HOME/ficc_python/clean_training_log.py $TRAINING_LOG_PATH

  # Unzip model and uploading it to automated training bucket
  MODEL_NAME='model'-${DATE_WITHOUT_YEAR}
  echo "Unzipping model $MODEL_NAME"
  gsutil cp -r gs://automated_training/model.zip $TRAINED_MODELS_PATH/model.zip
  unzip $TRAINED_MODELS_PATH/model.zip -d $TRAINED_MODELS_PATH/$MODEL_NAME
  if [ $? -ne 0 ]; then
    echo "Unzipping failed with exit code $?"
    /opt/conda/bin/python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Unzipping model failed. See attached logs for more details."
    exit 1
  fi

  echo "Uploading model to bucket"
  gsutil cp -r $TRAINED_MODELS_PATH/$MODEL_NAME gs://automated_training
  if [ $? -ne 0 ]; then
    echo "Uploading model to bucket failed with exit code $?"
    /opt/conda/bin/python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Uploading model to bucket failed. See attached logs for more details."
    exit 1
  fi
done
