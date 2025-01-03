# @ Author: Mitas Ray
# @ Create date: 2024-12-09
# @ Modified by: Mitas Ray
# @ Modified date: 2024-12-09
# @ Description: Authenticate Google Cloud service account using credentials JSON.

#!/bin/sh

# Set the path to your service account key JSON file
SERVICE_ACCOUNT_KEY="path/to/creds.json"

# Authenticate using the service account key
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"

# Verify authentication
echo "Authenticated as:"
gcloud auth list
