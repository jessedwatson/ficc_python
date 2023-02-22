'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-03-01 11:00:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-02-09 14:27:46
 # @ Description:
 '''

from google.cloud import storage
import pickle5 as pickle

'''
This function is used to upload data to the cloud bucket
'''

def upload_data(storage_client, bucket_name, file_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)
    print("File {} uploaded to {}.".format(file_name, bucket_name))

'''
This function is used to download the data from the GCP storage bucket.
It is assumed that we will be downloading a pickle file
'''

def download_data(storage_client, bucket_name, file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in) 
    print("File {} downloaded to {}.".format(file_name, bucket_name))
    return data