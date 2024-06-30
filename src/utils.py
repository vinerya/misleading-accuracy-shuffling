import numpy as np
from sklearn.utils import shuffle

def shuffle_data(X, y):
    return shuffle(X, y, random_state=None)
