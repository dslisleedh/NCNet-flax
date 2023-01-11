import os
from glob import glob
from natsort import natsorted
import shutil


def check_trained_model_exists():
    os.makedirs('./models/', exist_ok=True)

    if not os.path.exists('./models/trained/result.yaml'):
        print('No trained model found. Check if there is trained model in /logs')

        files = glob('./logs/*/result.yaml')
        if len(files) == 0:
            raise FileNotFoundError('No trained model found. Please train the model first.')
        else:
            files = natsorted(files, reverse=True)
            dir_name = '/'.join(files[0].split('/')[:-1])
            print('Found trained model in {}'.format(dir_name))
            shutil.copytree(dir_name, './models/trained/')


def check_quantization_model_exists():  # FIXME: not implemented
    return None
