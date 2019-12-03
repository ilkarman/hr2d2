import numpy as np
import torch
from git import Repo
from datetime import datetime
from os import path
from PIL import Image
from toolz import pipe
import os
import logging


def load_log_configuration(log_config_file):
    """
    Loads logging configuration from the given configuration file.
    """
    if not os.path.exists(log_config_file) or not os.path.isfile(log_config_file):
        msg = "%s configuration file does not exist!", log_config_file
        logging.getLogger(__name__).error(msg)
        raise ValueError(msg)
    try:
        logging.config.fileConfig(log_config_file, disable_existing_loggers=False)
        logging.getLogger(__name__).info("%s configuration file was loaded.", log_config_file)
    except Exception as e:
        logging.getLogger(__name__).error("Failed to load configuration from %s!", log_config_file)
        logging.getLogger(__name__).debug(str(e), exc_info=True)
        raise e




def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array, axis=0)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)

    array = torch.from_numpy(array)
    return array


def current_datetime():
    return datetime.now().strftime("%b%d_%H%M%S")


def git_branch():
    repo = Repo(search_parent_directories=True)
    return repo.active_branch.name


def git_hash():
    repo = Repo(search_parent_directories=True)
    return repo.active_branch.commit.hexsha


def generate_path(base_path, *directories):
    path = os.path.join(base_path, *directories)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _chw_to_hwc(image_array_numpy):
    return np.moveaxis(image_array_numpy, 0, -1)

