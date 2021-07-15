
import boto3 as bt
from pathlib import Path
import os.path






def upload_to_aws(path):
    file_name = os.path.basename(path)
    s3 = bt.client('s3', aws_access_key_id='xxxxxx',
                      aws_secret_access_key='xxxxxx')
    try:
        s3.upload_file(str(path), 'project2-models', file_name , ExtraArgs={'ACL': 'public-read'})
        return f'https://project2-models.s3.amazonaws.com/{file_name}'
    except FileNotFoundError:
        print("The file was not found")
        return None


def download_from_aws(file_name,destination_path):
    s3_client = bt.client('s3')
    s3 = bt.resource('s3',
                     aws_access_key_id='xxxxxxx',
                     aws_secret_access_key='xxxxxxxx')
    s3.Bucket('project2-models').download_file(file_name, destination_path)


#
# test = upload_to_aws('C:/Repos/model/APPL_model.h5')
# #
# #file = download_from_aws('APPL_model.h5','C:/Repos/model/APPL_model.h5')
# print(test)
# print('')