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
        self.rewards = []

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

        reward = self.chlorine_reward_shaped(sensor_readings=nodes_quality)
        self.rewards.append(reward)
        return reward

    def chlorine_reward(self, sensor_readings: np.ndarray,
                        target: float = 0.3,
                        sigma: float = 0.1,
                        lower: float = 0.2,
                        upper: float = 0.4,
                        scaling_factor : float = 1) -> float:
        """
        Compute rewards for a vector of chlorine sensor readings.

        Parameters:
            sensor_readings (np.ndarray): Array of sensor values.
            target (float): Ideal chlorine concentration.
            sigma (float): Std dev for Gaussian reward (within bounds).
            lower (float): Lower safe concentration bound.
            upper (float): Upper safe concentration bound.
            scaling_factor (float): Scaling factor for the final reward value.
        Returns:
            np.ndarray: Array of reward values.
        """

        rewards = np.zeros_like(sensor_readings)

        # Within bounds: Gaussian reward centered at target
        in_bounds = (sensor_readings >= lower) & (sensor_readings <= upper)
        rewards[in_bounds] = np.exp(-((sensor_readings[in_bounds] - target) ** 2) / (2 * sigma ** 2))

        # Out of bounds: Constant penalty when outside the bounds
        out_bounds = ~in_bounds
        rewards[out_bounds] = -1.0

        return scaling_factor * np.sum(rewards)

    def chlorine_reward_linear(self, sensor_readings: np.ndarray,
                        target: float = 0.3,
                        lower: float = 0.2,
                        upper: float = 0.4,
                        min_reward: float = -1.0,
                        scaling_factor: float = 1) -> float:
        """
        Compute rewards for a vector of chlorine sensor readings using a V-shaped
        piecewise linear function centered at `target`.

        Parameters:
            sensor_readings (np.ndarray): Array of sensor values.
            target (float): Ideal chlorine concentration.
            lower (float): Minimum chlorine level for linear reward range.
            upper (float): Maximum chlorine level for linear reward range.
            min_reward (float): Minimum reward at the bounds.
            scaling_factor (float): Scaling factor for the final reward value.

        Returns:
            float: Total reward (scaled).
        """

        rewards = np.zeros_like(sensor_readings)

        # Left side of the V (increasing towards target)
        left_mask = sensor_readings < target
        rewards[left_mask] = np.interp(sensor_readings[left_mask],
                                       [lower, target],
                                       [min_reward, 1.0])

        # Right side of the V (decreasing from target)
        right_mask = sensor_readings >= target
        rewards[right_mask] = np.interp(sensor_readings[right_mask],
                                        [target, upper],
                                        [1.0, min_reward])
        return scaling_factor * np.sum(rewards)

    def chlorine_reward_shaped(
            self,
            sensor_readings: np.ndarray,
            target: float = 0.3,
            sigma: float = 0.1,
            lower: float = 0.2,
            upper: float = 0.4,
            reward_scale: float = 1.0,
    ) -> float:
        """
        Shaped reward: average Gaussian‐style reward across nodes,
        minus a small cost on total injected chlorine.

        Parameters
        ----------
        sensor_readings : np.ndarray
            current Cl2 concentrations at each node
        target : float
            ideal concentration
        sigma : float
            stddev for the Gaussian “in‐bounds” shape
        lower, upper : float
            safe bounds (outside = constant penalty)
        reward_scale : float
            overall multiplier for the averaged node‐reward

        Returns
        -------
        float
            shaped reward in a roughly bounded range
        """
        # --- 1) node‐wise “accuracy” ---
        r = np.zeros_like(sensor_readings, dtype=float)

        in_bounds = (sensor_readings >= lower) & (sensor_readings <= upper)
        r[in_bounds] = np.exp(-((sensor_readings[in_bounds] - target) ** 2)
                              / (2 * sigma ** 2))
        r[~in_bounds] = -1.0

        avg_node_reward = np.mean(r)
        shaped = reward_scale * avg_node_reward
        return shaped
