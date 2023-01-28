import os
import sys
sys.path.append(os.getcwd())
import time
import pickle

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, MoehlisModelIntegrator, EsnIntegrator, EsnTrainer
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_moehlis_transition.extensions import RemotePythonTimeIntegrationGraph, LocalPythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, make_dump


def create_new_time_series(res, original_ts_filename, new_time_step):
    original_ts_path = os.path.join(res.local_research_path, original_ts_filename)
    with open(original_ts_path, 'rb') as f:
        original_training_ts = pickle.load(f)
    new_training_ts = {
        'time': original_training_ts['time'][::new_time_step],
        'timeseries': original_training_ts['timeseries'][::new_time_step],
    }
    new_training_ts_path = os.path.join(res.local_research_path, f'{original_ts_filename}_time_step_{new_time_step}')
    with open(new_training_ts_path, 'wb') as f:
        pickle.dump(new_training_ts, f)


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
#    ssh_comm = SshCommunication.create_from_config('atmlxint2')
#    res = Research.open('RC_MOEHLIS', ssh_comm)
    res = Research.open('RC_MOEHLIS')
    re = 275
    original_time_step = 1
    new_time_step = 10
    for ts_filename in (f'training_timeseries_re_{re}', f'test_timeseries_re_{re}'):
        create_new_time_series(res=res,
                               original_ts_filename=ts_filename,
                               new_time_step=new_time_step)
