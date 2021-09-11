import os


def is_in_docker():
    if os.path.exists('/docker'):
        return False
    else:
        return True
