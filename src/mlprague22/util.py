import os
import random
import numpy as np
import tensorflow as tf


RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mount_gdrive(path):
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        return f"/content/gdrive/MyDrive/{path}", True
    except:
        # init local CUDA device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "..", False