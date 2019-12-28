import os

def try_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass