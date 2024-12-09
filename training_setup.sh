# @ Author: Mitas Ray
# @ Create date: 2024-12-09
# @ Modified by: Mitas Ray
# @ Modified date: 2024-12-09

#!/bin/sh

MODEL_NAME="yield_spread_with_similar_trades"


# Define directories to create
DIRS="$HOME/training_logs $HOME/trained_models"

# Loop through each directory and create it if it doesn't exist
for DIR in $DIRS; do
  if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR" && echo "Created directory: $DIR" || echo "Failed to create directory: $DIR"
  else
    echo "Directory already exists: $DIR"
  fi
done


# Define the virtual environment directory and requirements file path
VENV_DIR="venv_py310"
REQUIREMENTS_DIR="$HOME/ficc_python"
REQUIREMENTS_FILE="$REQUIREMENTS_DIR/requirements_py310.txt"

# Check if the virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment '$VENV_DIR' already exists."
else
  # Create the virtual environment
  python3 -m venv "$VENV_DIR"
  
  if [ $? -eq 0 ]; then
    echo "Virtual environment '$VENV_DIR' created successfully."
  else
    echo "Failed to create the virtual environment."
    exit 1
  fi
fi

# Activate the virtual environment
. "$VENV_DIR/bin/activate"

if [ $? -eq 0 ]; then
  echo "Virtual environment '$VENV_DIR' activated."
else
  echo "Failed to activate the virtual environment."
  exit 1
fi

# Check if the requirements file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing packages from $REQUIREMENTS_FILE..."
  pip install -r "$REQUIREMENTS_FILE"
  
  if [ $? -eq 0 ]; then
    echo "Packages installed successfully."
  else
    echo "Failed to install packages. Check $REQUIREMENTS_FILE for errors."
    exit 1
  fi
else
  echo "Requirements file '$REQUIREMENTS_FILE' not found in $REQUIREMENTS_DIR. Skipping package installation."
fi

echo "Setup complete. Use 'source $VENV_DIR/bin/activate' to activate the virtual environment."


# Define the cron job
CRON_JOB="45 10 * * 1-5 sh /home/mitas/ficc_python/${MODEL_NAME}_deployment.sh >> /home/mitas/training_logs/${MODEL_NAME}_training_$(date +\%Y-\%m-\%d).log 2>&1"

# Check if the cron job already exists
crontab -l 2>/dev/null | grep -F "$CRON_JOB" >/dev/null

if [ $? -eq 0 ]; then
  echo "Cron job already exists. No changes made."
else
  # Add the cron job
  (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
  echo "Cron job added successfully."
fi
