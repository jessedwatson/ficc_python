'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2022-03-01
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-24
 # @ Description: Convenience functions to upload and download data from Google cloud buckets.
 '''
import pickle5 as pickle


def upload_data(storage_client, bucket_name, file_name):
    '''Upload data to the cloud bucket `bucket_name` with filename `file_name`.'''
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)
    print(f'File {file_name} uploaded to Google cloud bucket: {bucket_name}')


def download_data(storage_client, bucket_name, file_name):
    '''Download file `file_name` from the cloud bucket `bucket_name`. Assumes 
    that `file_name` is a pickle file.'''
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in) 
    print(f'File {file_name} downloaded from Google cloud bucket: {bucket_name}')
    return data
