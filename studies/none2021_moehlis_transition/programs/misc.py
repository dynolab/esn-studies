import os
import sys
import json

import numpy as np


def generate_random_perturbation(required_ke):
    rp = np.random.rand(9) - 0.5
    rp_ke = m.kinetic_energy(rp[np.newaxis, :])[0]
    norm_coeff = np.sqrt(rp_ke/required_ke)
    return rp / norm_coeff


def upload_paths_from_config():
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        conf = json.load(f)
    sys.path.append(conf['path_to_reducedmodels'])
    sys.path.append(conf['path_to_pyESN'])
