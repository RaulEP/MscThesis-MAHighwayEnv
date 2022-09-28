from turtle import width
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

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class MAHighwayEnv(AbstractEnv):    

    """
    add explanation here
    """
    CONTROLLED_VEHICLE_TYPE = "NULL"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "features": ["x", "y", "vx", "vy"],
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "speed_limit": 35,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "road_length": 1000,
            "vehicles_density": 1,
            "DLC_config": {
                "reward_speed_range": [20, 30],
                "weights": [5,10,1,1,1],
                        },
            "MLC_Config": {
                "reward_speed_range": [20, 30],
                "weights": [5,10,1,1,1]
                        }, 
            "normalize_reward": True,
            "offroad_terminal": True,
            "controlled_vehicle_types": ["highway_env.vehicle.controller.MLCVehicle", "highway_env.vehicle.controller.DLCVehicle"],
            "test_controlled": 0
        })
        return config
    
    #Temporal Classes
    def set_controlled_vehicle_class(self) -> None :
        self.CONTROLLED_VEHICLE_TYPE = utils.class_from_path(self.config["controlled_vehicle_types"][self.config['test_controlled']])
    
    def get_controlled_vehicle_class(self):
        return self.CONTROLLED_VEHICLE_TYPE

    def _reset(self) -> None:
        self._create_road()
        self.set_controlled_vehicle_class()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=self.config["speed_limit"], length=self.config["road_length"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        controlled_vehicle_types = self.get_controlled_vehicle_class()
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []

        vehicle = 0
        for others in other_per_controlled:
            if issubclass(controlled_vehicle_types, MLCVehicle):
                vehicle = controlled_vehicle_types.create_random(
                    road = self.road,
                    speed=25,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"], 
                )
                vehicle.MIN_SPEED = 15
                vehicle.MAX_SPEED = 30

            elif issubclass(controlled_vehicle_types, DLCVehicle):
                vehicle = controlled_vehicle_types.create_random(
                    self.road,
                    speed=35,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )
                vehicle.MIN_SPEED = 20
                vehicle.MAX_SPEED = 40

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
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
        sumReward = 0
        for key in rewards:
            sumReward += rewards[key][0] * rewards[key][1]

        return sumReward * float(self.vehicle.on_road)

    def _rewards(self, action: Action) -> Dict[Text, float]:
        
        #COMMON REWARDS

        collision_penalty = -1 if self.vehicle.crashed else 0
        lane_change_penalty = -1 if action in [0,2] else 0
        maintain_speed_range_reward = 0

        #CREATE FUNCTION THAT DETERMINES WHAT TYPE OF VEHICLE IT IS #REVISE

        controlled_vehicle_types = self.get_controlled_vehicle_class()
        if issubclass(controlled_vehicle_types, MLCVehicle):
            #MLC Reward Function
            if self.vehicle.lane_index[2] == 2:
                proactive_mlc_reward = (1 - (self.vehicle.position[0]/self.config["road_length"]))
            else:
                proactive_mlc_reward = self.vehicle.position[0]/self.config["road_length"]
            #ANALYZE THIS
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            scaled_speed = utils.lmap(forward_speed, self.config["MLC_Config"]["reward_speed_range"], [0, 1])
            
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            return {
                "proactive_mlc_reward": [proactive_mlc_reward, self.config["MLC_Config"]["weights"][0]],
                "collision_penalty": [collision_penalty, self.config["MLC_Config"]["weights"][1]],
                "lane change penalty": [lane_change_penalty ,self.config["MLC_Config"]["weights"][2]],
                "high_speed_reward": [np.clip(scaled_speed, 0, 1), self.config["MLC_Config"]["weights"][3]],
            }
        elif issubclass(controlled_vehicle_types, DLCVehicle):
            #DLC Reward FUNCTION
            if  self.vehicle.speed > self.vehicle.target_speed:
                target_speed_reward = 1
            else:
                target_speed_reward = 0

            return {
                    "collision_penalty": -1 if self.vehicle.crashed else 0,
                    "lane change penalty": -1 if action in [0,2] else 0 ,
                    "high_speed_reward": np.clip(scaled_speed, 0, 1),
                    #"~target speed reward": target_speed_reward,
                    "on_road_reward": float(self.vehicle.on_road)
            }
        

        
        
        
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

register(
    id='ma-highway-v0',
    entry_point='highway_env.envs:MAHighwayEnv',
)

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)