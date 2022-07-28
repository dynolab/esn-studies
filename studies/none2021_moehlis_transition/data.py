from typing import List, Union

from jsons import JsonSerializable


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    edge_tracking_simulations
      SimulationsInfo object with edge tracking simulations

    simulations_with_full_fields_saved
      SimulationsInfo object with simulations where full flow fields were saved with small time intervals (dT = 1 or 10)

    p_lam_info
      LaminarisationProbabilityInfo object with simulations associated with the estimation of the laminarisation
      probability
    """

    def __init__(self, edge_states_info: SimulationsInfo, simulations_with_full_fields_saved: SimulationsInfo,
                 p_lam_info: LaminarisationProbabilityInfo, seed: int, seed_for_bayesian_example: int,
                 default_sample_number: int, sample_size_per_energy_level: int,
                 minimum_sample_size_per_energy_level: int):

        self.res_id = res_id
        self.re = re
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.tasks = tasks
        self.task_for_uncontrolled_case = task_for_uncontrolled_case

        self.edge_states_info = edge_states_info
        self.simulations_with_full_fields_saved = simulations_with_full_fields_saved
        self.p_lam_info = p_lam_info
        self.seed = seed
        self.seed_for_bayesian_example = seed_for_bayesian_example
        self.default_sample_number = default_sample_number
        self.sample_size_per_energy_level = sample_size_per_energy_level
        self.minimum_sample_size_per_energy_level = minimum_sample_size_per_energy_level
