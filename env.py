"""
This module contains the water chlorination control environment that is to be used in the
"1st AI for Drinking Water Chlorination Challenge" @ IJCAI-2025.
"""
from typing import Optional, Any
import numpy as np
import math
from epyt_control.envs import EpanetMsxControlEnv
from epyt_control.envs.actions import SpeciesInjectionAction
from epyt_flow.simulation import ScadaData, SensorConfig, ScenarioConfig
from epyt_flow.utils import to_seconds


class WaterChlorinationEnv(EpanetMsxControlEnv):
    """
    Control environment.
    """

    def __init__(self, scenario_config: ScenarioConfig, f_in_contamination_metadata: str,
                 f_in_streams_data: str, action_space: list[SpeciesInjectionAction]):
        super().__init__(scenario_config=scenario_config,
                         action_space=action_space,
                         rerun_hydraulics_when_reset=False)

        self.__sensor_config_reward = None
        self._f_in_contamination_metadata = f_in_contamination_metadata
        self._f_in_streams_data = f_in_streams_data

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        # Reset
        super().reset(seed, options)

        # Set constant chlorine injection
        # self._scenario_sim.epanet_api.setMSXPattern("CL2PAT", [3000])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT1", [500])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT2", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT3", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT4", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT5", [10])

        # Skip first three days to give the network time to settle a proper initial state
        time_step = self._scenario_sim.epanet_api.getTimeHydraulicStep()
        n_steps_to_skip = int(to_seconds(days=3) / time_step)

        current_scada_data = None
        for _ in range(n_steps_to_skip):
            current_scada_data, _ = self._next_sim_itr()

        obs = self._get_observation(current_scada_data)

        return obs, {"scada_data": current_scada_data}

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).
        Sums up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        reward of zero means everythings is okay, while a negative reward denotes Cl concentration
        bound violations
        TODO: Override this method with a "better" reward function!

        Parameters
        ----------
        :class:`epyt_flow.simulation.ScadaData`
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        # TODO: Replace with smth. more reasonable!
        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everything is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0.

        # Regulation Limits (taken from the evaluation metrics)
        upper_cl_bound = .4  # (mg/l)
        lower_cl_bound = .2  # (mg/l)

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

        # reward_concentration = 0.
        # for cl_concentration in nodes_quality.flatten().tolist():
        #    reward_concentration += math.exp(-((cl_concentration - 0.3) ** 2) / (2 * 0.05 ** 2))

        # reward_concentration = reward_concentration / len(nodes_quality)

        # upper_bound_violation_idx = nodes_quality > upper_cl_bound
        # reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        # lower_bound_violation_idx = nodes_quality < lower_cl_bound
        # reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return self.chlorine_reward(sensor_readings=nodes_quality)

    def chlorine_reward(self, sensor_readings: np.ndarray,
                        target: float = 0.3,
                        sigma: float = 0.05,
                        alpha: float = 10.0,
                        lower: float = 0.2,
                        upper: float = 0.4) -> float:
        """
        Compute rewards for a vector of chlorine sensor readings.

        Parameters:
            sensor_readings (np.ndarray): Array of sensor values.
            target (float): Ideal chlorine concentration.
            sigma (float): Std dev for Gaussian reward (within bounds).
            alpha (float): Penalty factor for out-of-bound values.
            lower (float): Lower safe concentration bound.
            upper (float): Upper safe concentration bound.

        Returns:
            np.ndarray: Array of reward values.
        """

        rewards = np.zeros_like(sensor_readings)

        # Within bounds: Gaussian reward centered at target
        in_bounds = (sensor_readings >= lower) & (sensor_readings <= upper)
        rewards[in_bounds] = np.exp(-((sensor_readings[in_bounds] - target) ** 2) / (2 * sigma ** 2))

        # Out of bounds: Linear penalty based on distance to nearest bound
        out_bounds = ~in_bounds
        distances = np.where(sensor_readings[out_bounds] < lower,
                             lower - sensor_readings[out_bounds],
                             sensor_readings[out_bounds] - upper)
        rewards[out_bounds] = -alpha * distances

        return np.sum(rewards)
