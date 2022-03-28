import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

if not os.path.isdir('data'):
    os.mkdir('data')

if os.path.isfile(os.path.join('data', 'Test.csv')):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_cli('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign')

    zip_path = os.path.join(api.get_default_download_dir(), 'gtsrb-german-traffic-sign.zip')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
