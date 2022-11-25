from typing import Dict, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MLCVehicle, DLCVehicle
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import math

Observation = np.ndarray

class MAHighwayEnv(AbstractEnv):    

    """
    add explanation here
    """
    vehicle_crashed = False
    vehicles_speed = []

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "normalize": True,
                    "features": ["x", "y", "vx", "vy"]}},
            "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
            }},
            "lanes_count": 3,
            "initial_lane_id": None,
            "speed_limit": 33,
            "duration": 40,  # [s]
            "simulation_frequency": 60,  # [Hz]
            "policy_frequency": 5,  # [Hz]
            "ego_spacing": 1,
            "road_length": 1000,
            "screen_width": 1800, 
            "screen_height": 150, 
            "centering_position": [0.3, 0.5], 
            "scaling": 5.0,
            "vehicles_density": 1,
            "normalization_range": [-100, 4],
                    "DLC_config": {
                        "count": 8,
                        "reward_speed_range": [23, 28], #speed range should be bellow target speed > 28.
                        "weights": [1,133,1,1,2,1],
                            },
                    "MLC_config": {
                        "count": 8,
                        "reward_speed_range": [19, 23],
                        "weights": [1,133,1,1,1,2],
                            },

            "normalize_reward": False,
            "offroad_terminal": True,
            "controlled_vehicle_types": ["highway_env.vehicle.controller.MLCVehicle", "highway_env.vehicle.controller.DLCVehicle"],
            "test_controlled": 0
        })
        return config

    def _reset(self) -> None:
        self.vehicle_crashed = False
        self.vehicles_speed = []
        self._create_road()
        self._create_vehicles()
        for i in range(len(self.controlled_vehicles)):
            self.vehicles_speed.append(self.controlled_vehicles[i].speed)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=self.config["speed_limit"], length=self.config["road_length"]),
                            np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        controlled_vehicle_types = self.config["controlled_vehicle_types"]
        vehicle_type_one = utils.class_from_path(controlled_vehicle_types[0])
        vehicle_type_two = utils.class_from_path(controlled_vehicle_types[1])
        type_one_per_type_two = near_split(self.config["DLC_config"]["count"], num_bins=self.config["MLC_config"]["count"])

        self.controlled_vehicles = []

        vehicle = 0
        for others in type_one_per_type_two:
            vehicle = vehicle_type_one.create_random(
                road = self.road,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"], 
            )
            
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = vehicle_type_two.create_random(
                    self.road,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )

                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        
        return super().step(action)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        vehicles_rewards = []
        vehicle_id = 0
        for vehicle in rewards:
            sumReward = 0
            for key in vehicle:
                sumReward += vehicle[key][0] * vehicle[key][1]
            if self.config["normalize_reward"]:
                sumReward = utils.lmap(sumReward, self.config["normalization_range"], [0, 1])
            total_reward = sumReward * float(self.controlled_vehicles[vehicle_id].on_road)
            vehicles_rewards.append(total_reward)
            vehicle_id += 1
        average_reward = np.average(vehicles_rewards)
        return average_reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        
        #COMMON REWARDS
        vehicle_id = 0
        time_headway_threshold = 1
        controlled_vehicle_rewards = []
        self.vehicles_speed = []
        for i in range(len(self.controlled_vehicles)):
            self.vehicles_speed.append(self.controlled_vehicles[i].speed)

        for v_action in action:
                front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicles[vehicle_id])
                time_headway_reward = 0
                if not(front_vehicle is None):                    
                    front_vehicle_distance = self.controlled_vehicles[vehicle_id].front_distance_to(front_vehicle)
                    if (front_vehicle_distance <= 100) and (front_vehicle_distance > self.controlled_vehicles[vehicle_id].speed):
                        time_hr = front_vehicle_distance/(self.controlled_vehicles[vehicle_id].speed*time_headway_threshold)
                        time_headway_reward = utils.lmap(time_hr, [0,5.88], [0, 1])
                collision_penalty = -1 if self.controlled_vehicles[vehicle_id].crashed else 0
                lane_change_penalty = -1 if v_action in [0,2] else 0
                v_class = type(self.controlled_vehicles[vehicle_id])
                if issubclass(v_class, MLCVehicle):
                    #MLC Reward Function
                    if self.controlled_vehicles[vehicle_id].lane_index[2] == 2:
                        proactive_mlc_reward = (1 - (self.controlled_vehicles[vehicle_id].position[0]/self.config["road_length"]))
                        target_lane_reward = 1
                    else:
                        proactive_mlc_reward = -self.controlled_vehicles[vehicle_id].position[0]/self.config["road_length"]
                        target_lane_reward = 0
                    

                    #ANALYZE THIS
                    forward_speed = self.controlled_vehicles[vehicle_id].speed * np.cos(self.controlled_vehicles[vehicle_id].heading)
                    scaled_speed = utils.lmap(forward_speed, self.config["MLC_config"]["reward_speed_range"], [0, 1])
                    
                    # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
                    controlled_vehicle_rewards.append(

                        {
                        "time_headway_reward": [time_headway_reward, self.config["MLC_config"]["weights"][0]],    
                        "collision_penalty": [collision_penalty, self.config["MLC_config"]["weights"][1]],
                        "lane change penalty": [lane_change_penalty ,self.config["MLC_config"]["weights"][2]],
                        "speed_range_reward": [np.clip(scaled_speed, 0, 1), self.config["MLC_config"]["weights"][3]],
                        "target_lane_reward": [target_lane_reward, self.config["MLC_config"]["weights"][4]], #create in config
                        "proactive_mlc_reward": [proactive_mlc_reward, self.config["MLC_config"]["weights"][5]],
                        
                        })

                elif issubclass(v_class, DLCVehicle):
                    #DLC Reward FUNCTION
                    if  self.controlled_vehicles[vehicle_id].speed > self.controlled_vehicles[vehicle_id].target_speed:
                        target_speed_reward = 1
                    else:
                        target_speed_reward = (self.controlled_vehicles[vehicle_id].speed - self.config["DLC_config"]["reward_speed_range"][0]) \
                                /(self.config["DLC_config"]["reward_speed_range"][1] - self.config["DLC_config"]["reward_speed_range"][0])

                    if self.controlled_vehicles[vehicle_id].destination[0] >= 10000:
                        finish_road_reward = 1
                    else:
                        finish_road_reward = self.controlled_vehicles[vehicle_id].destination[0]/10000

                    #ANALYZE THIS    
                    forward_speed = self.controlled_vehicles[vehicle_id].speed * np.cos(self.controlled_vehicles[vehicle_id].heading)
                    scaled_speed = utils.lmap(forward_speed, self.config["DLC_config"]["reward_speed_range"], [0, 1])

                    controlled_vehicle_rewards.append(

                        {
                        "time_headway_reward": [time_headway_reward, self.config["MLC_config"]["weights"][0]],    
                        "collision_penalty": [collision_penalty, self.config["MLC_config"]["weights"][1]],
                        "lane change penalty": [lane_change_penalty ,self.config["MLC_config"]["weights"][2]],
                        "speed_range_reward": [np.clip(scaled_speed, 0, 1), self.config["MLC_config"]["weights"][3]],
                        "target_speed_reward": [target_speed_reward, self.config["DLC_config"]["weights"][4]],
                        "finish_road_reward": [finish_road_reward, self.config["DLC_config"]["weights"][5]],
                        })
                
                vehicle_id += 1

        return controlled_vehicle_rewards


    def _info(self, obs: Observation, action: Action) -> dict:
        
        odd = [1,3,5,7,9,11,13,15] 
        info = {
            "vehicles_speed": self.vehicles_speed,
            "avg_speed": sum(self.vehicles_speed)/len(self.vehicles_speed),
            "avg_mlc_speed": sum([self.vehicles_speed[mlc] for mlc in range(len(self.vehicles_speed)) if mlc % 2 == 0 or mlc == 0])/(len(self.vehicles_speed)/2),
            "avg_dlc_speed": sum([self.vehicles_speed[dlc] for dlc in odd])/(len(self.vehicles_speed)/2),
            "crashed": self.vehicle_crashed,
            "action": action,
        }

        try:
            info["rewards"] = self._rewards(action)

        except NotImplementedError:
            pass
        return info

                
    def _is_terminated(self) -> bool:
        """
        The episode is over if any of the vehicle crashes or its outside of road
        """
        for i in range(len(self.controlled_vehicles)):
            if self.controlled_vehicles[i].crashed or not self.controlled_vehicles[i].on_road:
                self.vehicle_crashed = True
                return True
        return False

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

register(
    id='ma-highway-v0',
    entry_point='highway_env.envs:MAHighwayEnv',
)