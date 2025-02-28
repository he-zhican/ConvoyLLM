import math
import os
import random
import cv2
import gym
import numpy as np
from gym import spaces
import traci
from sumo.convoyVehicle import ConvoyVehicle
from sumo.simpleStruct import Road
from sumo.vehicle import Vehicle


class MutilEnv(gym.Env):
    def __init__(self, render_mode=None,sumo_home="", task=0, result_folder=""):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.render_mode = render_mode
        self.task = task
        self.result_folder = result_folder
        # os.environ['SUMO_HOME'] = '/usr/share/sumo'
        # Define the SUMO executable path
        self.client = traci
        if render_mode == "human":
            self.sumo_binary = sumo_home+"sumo-gui"  # If you want to use GUI mode, use "sumo-gui", otherwise use "sumo"
        else:
            self.sumo_binary = sumo_home+"sumo"
        # A SUMO configuration file containing map and simulation parameters
        self.sumo_config = f"configs/convoy_{self.task}.sumocfg"
        self.client.start([self.sumo_binary, "-c", self.sumo_config, "--start"])
        self.cv_names = ["veh1", "veh2", "veh3", "veh4", "veh5", "veh6", "veh7", "veh8"]
        self.desired_lanes = [1, 0, 1, 0, 1, 0, 1, 0]
        self.aver_speeds = [0] * 8  # The average speed of all convoy vehicles
        self.convoy_vehicles = []  # All convoy vehicles
        self.task_vehicle: ConvoyVehicle = None
        # Each time a decision is made by the LLM, the environment executes 10 time steps
        self.decision_frequency = 10
        self.dt = 0.025
        self.video_writer = None
        # The width and height of the video frame
        self.width = 1148
        self.height = 200

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.convoy_vehicles.clear()
        self.video_writer = None
        # reload the environment
        self.client.load(
            ["-c", f"configs/convoy_{self.task}.sumocfg", "--start", "--seed", f"{seed}", "--lanechange.duration",
             "1.5"])
        self.client.simulationStep()

        result_files = [self.result_folder + '/speed.txt', self.result_folder + '/position_error.txt']

        for file_path in result_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Initialize all convoy vehicles
        self.convoy_vehicles = self.get_all_cvs()
        # The perspective is focused on veh5
        if self.render_mode == "human":
            if self.task == 0:
                self.client.gui.trackVehicle("View #0", "veh5")
            else:
                self.client.gui.trackVehicle("View #0", self.task_vehicle.id)
            self.client.gui.setZoom("View #0", 580)
        # Update the states of the self and environment vehicles
        states = self.get_state()
        return states, {}

    def step(self, actions):
        # Action Decoder
        for i, cv in enumerate(self.convoy_vehicles):
            action = self.get_avail_actions(cv, actions[i])
            if action == 1:  # maintain
                # cv.target_lane = cv.lane
                pass
            elif action == 0:  # turn left
                cv.target_lane = cv.lane + 1
            elif action == 2:  # turn right
                cv.target_lane = cv.lane - 1
            elif action == 3:  # accelerate
                cv.target_speed = cv.last_target_speed + 1.0
            elif action == 4:  # decelerate
                cv.target_speed = cv.last_target_speed - 2.0

        states = []
        rewards = []
        dones = []
        truncateds = []
        for _ in range(self.decision_frequency):
            self.write_data()
            states.clear()
            rewards.clear()
            dones.clear()
            truncateds.clear()
            for i, cv in enumerate(self.convoy_vehicles):
                reward, done, truncated = self.get_reward(cv)
                rewards.append(reward)
                dones.append(done)
                truncateds.append(truncated)
                cv.planning(self.convoy_vehicles)
                self.client.vehicle.moveToXY(cv.id, "", cv.lane, cv.new_x, cv.new_y,
                                             angle=90 - cv.heading * 180 / math.pi, keepRoute=2)
                self.aver_speeds[i] = round((self.aver_speeds[i] + cv.speed) / 2, 2)
            self.client.simulationStep()
            states = self.get_state()
            if True in dones or True in truncateds:
                break

            if self.render_mode == "human":
                self.render()

        return states, rewards, dones, truncateds, {}

    def get_state(self):
        # Update the environment vehicles
        all_evs = self.get_all_evs()
        # Update the convoy vehicles, their neighborhoods and surround_vehicles
        for cv in self.convoy_vehicles:
            cv.update_state(self.client)
            cv.find_surround_evs(all_evs)
            cv.find_neighborhoods(self.convoy_vehicles)

        return []

    def get_reward(self, convoy_vehicle):
        done = False  # Termination due to collision
        truncated = False  # Termination caused by the convoy vehicle driving the entire road
        reward = 0

        for cv in convoy_vehicle.neighborhoods:
            if cv is not None:
                # collision with neighborhoods
                if (abs(convoy_vehicle.x - cv.x) <= (convoy_vehicle.length + cv.length) / 2
                        and abs(convoy_vehicle.y - cv.y) <= (convoy_vehicle.width + cv.width) / 2):
                    done = True
                    break

        for ev in convoy_vehicle.surround_evs:
            if ev is not None:
                # collision with environment vehicles
                if (abs(convoy_vehicle.x - ev.x) <= (convoy_vehicle.length + ev.length) / 2
                        and abs(convoy_vehicle.y - ev.y) <= (convoy_vehicle.width + ev.width) / 2):
                    done = True
                    break

        # Leave the road (at this point the vehicle has traveled the entire distance)
        if convoy_vehicle.x < 0 or convoy_vehicle.x >= 980:
            truncated = True

        # The task is completed when the task-vehicle reaches a safe distance from the convoy vehicles
        if self.task == 1 and self.task_vehicle is not None:
            position_errors = self.get_position_error()
            if self.task_vehicle.neighborhoods[2] is not None:
                if all(abs(position_error) < 0.01 for position_error in position_errors):
                    truncated = True
                    self.task_vehicle.state = 0
                    if self.render_mode == "human":
                        self.client.vehicle.setColor(self.task_vehicle.id, (255, 255, 0))
            elif self.task_vehicle.neighborhoods[3] is not None:
                if all(abs(position_error) < 0.01 for position_error in position_errors):
                    truncated = True
                    self.task_vehicle.state = 0
                    if self.render_mode == "human":
                        self.client.vehicle.setColor(self.task_vehicle.id, (255, 255, 0))

        # the task is completed when the task-vehicle is out of range of all other convoy vehicles
        if self.task == 2 and self.task_vehicle is not None:
            # All neighbors of the task-vehicle are None,
            # indicating that they are out of the convoy communication range at this time
            if all(neighbor is None for neighbor in self.task_vehicle.neighborhoods):
                truncated = True
                self.task_vehicle.state = 4

        # If it is a mission that is switched to an escort formation,
        # the task is completed when the position error of all vehicles is less than 0.01
        if self.task == 3:
            position_errors = self.get_position_error()
            if (all(abs(PE) < 0.01 for PE in position_errors)
                    and all(cv.lane == cv.desired_lane for cv in self.convoy_vehicles)):
                truncated = True

        return reward, done, truncated

    # Calculate the formation position error for each convoy vehicle
    def get_position_error(self):
        position_errors = []
        for cv in self.convoy_vehicles:
            position_error = 0
            for i, neighbor in enumerate(cv.neighborhoods):
                if neighbor is not None:
                    dis = abs(Road.relation_distance(cv.x, neighbor.x))
                    if i == 2 or i == 3:
                        position_error += round(abs(dis - cv.safe_distance), 2)
                    else:
                        position_error += round(abs(dis - cv.safe_distance / 2), 2)
            position_errors.append(position_error)
        return position_errors

    def write_data(self):
        with open(self.result_folder + '/speed.txt', 'a') as file:
            for cv in self.convoy_vehicles:
                file.write(f'{round(cv.speed, 2)},')
            file.write('\n')
        with open(self.result_folder + '/position_error.txt', 'a') as file:
            position_errors = self.get_position_error()
            for position_error in position_errors:
                file.write(f'{round(position_error, 2)},')
            file.write('\n')

    def get_avail_actions(self, cv, action):
        if (cv.lane == 2 and action == 0) or (cv.lane == 0 and action == 2):
            action = 1
        elif (cv.speed <= 1 and action == 4) or (cv.speed >= 30 and action == 3):
            action = 1

        return action

    # Get all convoy vehicles
    def get_all_cvs(self):
        all_cvs = []
        for i, cv_id in enumerate(self.cv_names):
            temp_v = ConvoyVehicle(cv_id, self.desired_lanes[i])
            temp_v.update_state(self.client)
            self.aver_speeds[i] = temp_v.speed  # Initialize the average speed of each vehicle
            all_cvs.append(temp_v)

        # For the joining task, specify Vehicle veh1 as the vehicle to join the Fleet
        if self.task == 1:
            self.task_vehicle = all_cvs[0]
            self.task_vehicle.state = self.task
            if self.render_mode == "human":  # Change to blue
                self.client.vehicle.setColor(self.task_vehicle.id, (0, 100, 250))
        # For the leaving task, one of the 8 vehicles is randomly selected as the Detachment Vehicle
        if self.task == 2:
            # A vehicle is randomly selected as the vehicle to be leaving from the convoy
            task_cv = random.randint(0, 7)
            self.task_vehicle = all_cvs[task_cv]
            self.task_vehicle.state = self.task
            if self.render_mode == "human":
                self.client.vehicle.setColor(self.task_vehicle.id, (0, 100, 255))
        if self.task == 3:
            convoy_desired_lanes = [1, 0, 2, 1, 2, 0, 1, 1]
            self.task_vehicle = all_cvs[3]  # veh4
            if self.render_mode == "human":
                self.client.vehicle.setColor(self.task_vehicle.id, (0, 100, 255))
            for cv in all_cvs:
                cv.desired_lane = convoy_desired_lanes[all_cvs.index(cv)]
                cv.state = self.task
        return all_cvs

    # Get all the environment vehicles in SUMO
    def get_all_evs(self):
        all_evs = []
        vehicles_list = self.client.vehicle.getIDList()
        for c in vehicles_list:
            if c not in self.cv_names:
                temp_v = Vehicle(c)
                temp_v.update_state(self.client)
                all_evs.append(temp_v)
        return all_evs

    def get_all_lead_cvs(self):
        all_lead_cvs = []
        # Get the lead vehicles in the convoy and task-vehicle
        for cv in self.convoy_vehicles:
            if (cv.find_neighborhoods(self.convoy_vehicles)[2] is None
                    or cv.state == 1 or cv.state == 2 or cv.state == 3):
                all_lead_cvs.append(cv)
        return all_lead_cvs

    def render(self):
        screenshot_filename = f"screenshots/screenshot.png"
        # Call the traci.gui.screenshot method to take a screenshot
        self.client.gui.screenshot("View #0", screenshot_filename, width=self.width, height=self.height)
        # Read the picture and write it to the video file
        img = cv2.imread(screenshot_filename)
        if img is not None:
            self.video_writer.write(img)

    # Initialize the video writer
    def initialize_video_writer(self, video_path):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V encoding
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (self.width, self.height))

    def close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()

    def get_aver_speeds(self):
        return self.aver_speeds

    def close(self):
        self.client.close()
