import numpy as np
import os.path


def get_includes():
    root = os.path.dirname(__file__)
    path = [root, np.get_include()]
    return path

