import pandas as pd
import shutil
import time
import os


def save_csv(loss, csv_file=''):

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Epoch', 'Loss'])

    df = df.dropna(axis=1, how='all')

    new_data = pd.DataFrame({'Epoch': [len(df) + 1], 'Loss': [loss]})

    df = pd.concat([df, new_data], ignore_index=True)

    df.to_csv(csv_file, index=False)


def clearDir(dir_path=''):
    if not dir_path:
        print("pls fill in dir path.")
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Function '{}' took {:.4f} seconds to execute.".format( func.__name__, (end_time - start_time)))
        return result
    return wrapper