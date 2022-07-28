# Papers

Each paper we publish or plan to publish has the associated directory in `/papers` where codes for data abstractions, processing and plotting are stored. Here is a general structure of paper data:
```
papers
  |
  ------ paper_name_1
  |             |
  |             ----------- launchers
  |             |               |
  |             |               ----------- launcher_1.py
  |             |               |
  |             |               ----------- launcher_2.py
  |             |               |
  |             |               ----------- launcher_3.py
  |             |
  |             ----------- programs
  |             |               |
  |             |               ----------- program_1.py
  |             |               |
  |             |               ----------- program_2.py
  |             |               |
  |             |               ----------- program_3.py
  |             |
  |             ----------- views
  |             |             |
  |             |             ----------- figure_1.py
  |             |             |
  |             |             ----------- figure_2.py
  |             |             |
  |             |             ----------- figure_3.py
  |             |
  |             ----------- data.py
  |             |
  |             ----------- extensions.py
  |
  ------ paper_name_2
  |             |
  |             ----------- launchers
  |             |
  |             ----------- programs
  |             |
  |             ----------- views
  |             |
  |             ----------- data.py
  |             |
  |             ----------- extensions.py
  |
```
Let's describe this structure:
* `.../launchers`: each `.py` file in this directory can be used to launch a particular task either on a local or remote machine
* `.../programs`: each file in this directory is an executable (or a self-contained `.py` script) which can be used as a part of a task/algorithm/graph
* `.../views`: each file in this directory can be used to produce a figure (some of these figures can be found in the paper, some of them are exploratory)
* `.../data.py`: summary data representation implemented as `class Summary`; instances of this class are json-serializable and stored in `/jsons` 
* `.../extensions.py`: any other codes necessary to process research data (codes from this directory can migrate to `/restools`)


## Paper none2021_predicting_transition_using_reservoir_computing

A. Pershin, C. Beaume, K. Li, S. M. Tobias. Can neural networks predict dynamics they have never seen? (2021) [arXiv](https://arxiv.org/abs/2111.06783) 

### Launchers

:white_check_mark: time_integrate_ensemble_esn.py

Runs an ensemble of ESN simulations whose initial conditions (namely, 10 initial states for each simulation) are taken from some source task already containing time series (for example, computed via time-integrating Moehlis model). To make a graph, `LocalPythonTimeIntegrationGraph` (for local computations) or `RemotePythonTimeIntegrationGraph` (for remote computations) with `restools.standardised_programs.EsnIntegrator` is used.

:white_check_mark: time_integrate_ensemble_esn_for_probabilistic_prediction.py

Same as `time_integrate_ensemble_esn.py`, but initial conditions are specifically chosen to make probabilistic predictions for a given time series.

:white_check_mark: time_integrate_ensemble_esn_for_laminarization_probability.py

Same as `time_integrate_ensemble_esn.py`, but initial conditions are specifically chosen (these are small-amplitude perturbations from the laminar state, already used for Moehlis model) to get trajectories from which the laminarization probability can be computed.

:white_check_mark: time_integrate_ensemble_moehlis.py

Runs an ensemble of Moehlis model simulations whose initial conditions are randomly generated using formula for lifetime distributions: `apers.none2021_predicting_transition_using_reservoir_computing.extensions.generate_random_ic_for_lifetime_distr`. To make a graph, `LocalPythonTimeIntegrationGraph` (for local computations) or `RemotePythonTimeIntegrationGraph` (for remote computations) with `restools.standardised_programs.MoehlisModelIntegrator` is used.

:white_check_mark: time_integrate_ensemble_moehlis_for_laminarization_probability.py

Same as `time_integrate_ensemble_moehlis.py`, but a different formula was used for random initial conditions: `apers.none2021_predicting_transition_using_reservoir_computing.extensions.generate_random_ic_for_laminarization_probability` 



