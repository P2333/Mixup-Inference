import os
import numpy as np


def save_np_arrays(name, out_folder=None, **kwargs):
    if out_folder is None:
        raise ValueError("None type")
    out_path = os.path.join(out_folder, name)
    np.savez(out_path, **kwargs)
