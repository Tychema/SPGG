import os

def mkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)