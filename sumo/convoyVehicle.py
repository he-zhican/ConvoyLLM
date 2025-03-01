import math
import numpy as np

from sumo.vehicle import Vehicle
from sumo.simpleStruct import TIMESTEP, Direction, Road


class ConvoyVehicle(Vehicle):
    def __init__(self, vehicle_id, desired_lane=-1, desired_speed=25, state=0):
        super().__init__(vehicle_id)
        # [avoiding obstacles, joining, leaving, Switching to escort formation, leave successfully]<-->[0,1,2,3,4]
        self.state = state
        self.target_lane = desired_lane
        self.desired_lane = desired_lane
        self.desired_speed = desired_speed
        self.target_speed = desired_speed
        # the target speed in the last time
        self.last_target_speed = self.target_speed

        # The default safety distance between the front and rear
        # Vehicle is 4 meters long and the actual distance between the front and rear is 6 meters
        self.safe_distance = 10
        self.communication_distance = 60  # The maximum communication distance between vehicles
        self.perceive_distance = 150  # The maximum distance of the vehicle that perceives the environment vehicles

        self.old_G_force = 0  # 图的虚拟力
        self.new_G_force = 0
        self.old_L_force = 25.0  # 势场的纵向虚拟力
        self.new_L_force = 25.0
        self.old_H_force = 0  # 势场的横向虚拟力
        self.new_H_force = 0
        self.new_x = 0
        self.new_y = 0
        self.old_sita = 0

        # Stores information about neighbor-vehicles
        # [right_front_v,right_behind_v,front_v,behind_v,left_front_v,left_behind_v]
        self.neighborhoods = [None] * 6
        self.surround_evs = []

    def update_state(self, client):
        super().update_state(client)
        self.new_G_force = 0
        self.neighborhoods = [None] * 6
        self.surround_evs.clear()
        self.new_L_force = self.target_speed

    def show_state(self):
        super().show_state()
        print(f"target_lane:{self.target_lane}")
        print(f"desired_lane:{self.desired_lane}")

    def find_neighborhoods(self, c_vehicles):
        # When the ego-vehicle successfully leaving from the convoy, the neighbor node information is no longer obtained
        if self.state == 4:
            return self.neighborhoods
        for cv in c_vehicles:
            if cv.id == self.id or abs(cv.target_lane - self.target_lane) > 1 or cv.state == 4:
                continue
            if cv.target_lane == cv.lane:
                # Corresponding neighborhoods subscript[0,2,4]
                lane_index = int((cv.target_lane - self.target_lane + 1) * 2)
            else:
                if cv.target_lane == self.target_lane or cv.lane == self.target_lane:
                    lane_index = 2
                else:
                    continue
            front_v = self.neighborhoods[lane_index]
            behind_v = self.neighborhoods[lane_index + 1]
            dis = Road.relation_distance(self.x, cv.x)
            # If there are environment vehicles between the convoy vehicles, they are not considered neighbor nodes
            front_sv_dis = self.communication_distance
            behind_sv_dis = -self.communication_distance
            # Calculates the distance between the ego and the nearest environment vehicle in front and behind the same lane
            for sv in self.surround_evs:
                # The environment vehicle may collide with the vehicle regardless of whether
                # it is in the current lane or the target lane of the vehicle, so both situations need to be considered
                if sv is not None and (sv.lane == self.target_lane or sv.lane == self.lane):
                    dis_sv = Road.relation_distance(self.x, sv.x)
                    if (dis_sv > 0 and front_sv_dis == self.communication_distance) or 0 < dis_sv < front_sv_dis:
                        front_sv_dis = dis_sv
                    elif (dis_sv < 0 and behind_sv_dis == -self.communication_distance) or behind_sv_dis < dis_sv < 0:
                        behind_sv_dis = dis_sv
            if 0 <= dis <= front_sv_dis:
                if front_v is None or dis < Road.relation_distance(self.x, front_v.x):
                    front_v = cv
            elif behind_sv_dis <= dis < 0:
                if behind_v is None or dis > Road.relation_distance(self.x, behind_v.x):
                    behind_v = cv

            self.neighborhoods[lane_index] = front_v
            self.neighborhoods[lane_index + 1] = behind_v

        # if there is N_f between ego and N_fj, then N_fj is not considered as a neighbor node
        if self.neighborhoods[2] is not None:
            dis1 = Road.relation_distance(self.x, self.neighborhoods[2].x)
            if self.neighborhoods[0] is not None:
                dis2 = Road.relation_distance(self.x, self.neighborhoods[0].x)
                if dis1 < dis2:
                    self.neighborhoods[0] = None
            if self.neighborhoods[4] is not None:
                dis2 = Road.relation_distance(self.x, self.neighborhoods[4].x)
                if dis1 < dis2:
                    self.neighborhoods[4] = None
        # if there is N_b between ego and N_bj, then N_bj is not considered as a neighbor node
        if self.neighborhoods[3] is not None:
            dis1 = Road.relation_distance(self.x, self.neighborhoods[3].x)
            if self.neighborhoods[1] is not None:
                dis2 = Road.relation_distance(self.x, self.neighborhoods[1].x)
                if dis1 > dis2:
                    self.neighborhoods[1] = None
            if self.neighborhoods[5] is not None:
                dis2 = Road.relation_distance(self.x, self.neighborhoods[5].x)
                if dis1 > dis2:
                    self.neighborhoods[5] = None
        return self.neighborhoods

    # Find the nearest n environment vehicles within the self-observation distance
    def find_surround_evs(self, e_vehicles, n=10):
        min_distance = self.perceive_distance
        surround_evs = []
        for vehicle in e_vehicles:
            dis = Road.relation_distance(self.x, vehicle.x)
            if abs(dis) <= min_distance:
                surround_evs.append(vehicle)

        # Sort by distance in ascending order
        surround_evs.sort(key=lambda ev: abs(Road.relation_distance(self.x, ev.x)))

        while len(surround_evs) < n:
            surround_evs.append(None)

        self.surround_evs = surround_evs[:n]
        return self.surround_evs

    def graph_force(self):
        count = 0
        for i, neighborhood in enumerate(self.neighborhoods):
            if neighborhood is None:
                continue
            if ((neighborhood.state == 1 or neighborhood.state == 2)
                    and (neighborhood.neighborhoods[2] is None or neighborhood.neighborhoods[3] is None)):
                continue
            dis = Road.relation_distance(self.x, neighborhood.x)
            err_dis = 0
            if i == 2:  # front_v
                err_dis += 2 * (dis - self.safe_distance)
            elif i == 3:  # behind_v
                err_dis += 1 * (dis + self.safe_distance)
            elif i == 0 or i == 4:  # other_lane_front_v
                err_dis += 1 * (dis - self.safe_distance / 2)
            else:  # other_lane_behind_v
                err_dis += 0.1 * (dis + self.safe_distance / 2)
                pass
            count += 1
            self.new_G_force += err_dis

        if count > 0:
            self.new_G_force /= count

        if (self.new_G_force - self.old_G_force) / TIMESTEP > 2:
            self.new_G_force = self.old_G_force + 2 * TIMESTEP
        if (self.new_G_force - self.old_G_force) / TIMESTEP < -4.0:
            self.new_G_force = self.old_G_force - 4.0 * TIMESTEP
        if self.new_G_force > 5:
            self.new_G_force = (self.new_G_force / abs(self.new_G_force)) * 5
        if self.new_G_force < -self.desired_speed:
            self.new_G_force = (self.new_G_force / abs(self.new_G_force)) * self.desired_speed

    def keep_lane_force(self):
        left_distance, right_distance = Road.get_lanes_distance(self.y, self.lane)

        half_lane_width = Road.lane_width / 2

        if left_distance < half_lane_width:
            keep_force = 1.8 - left_distance
            self.new_H_force = -1.8 * keep_force

        if right_distance < half_lane_width:
            keep_force = 1.8 - right_distance
            self.new_H_force = 1.8 * keep_force

    def change_lane_force(self):
        left_distance, right_distance = Road.get_lanes_distance(self.y, self.lane)

        # The target lane is to the left of the current lane
        if self.target_lane > self.lane:
            self.new_H_force = 1.8 * (2.0 - left_distance)
            if self.old_H_force > self.new_H_force:
                self.new_H_force = self.old_H_force
        # The target lane is to the right of the current lane
        if self.target_lane < self.lane:
            self.new_H_force = 1.8 * (-2.0 + right_distance)
            if self.old_H_force < self.new_H_force:
                self.new_H_force = self.old_H_force

    def collision_avoid(self, vehicles):
        # If the vehicle was supposed to change lanes, but found that the distance between the vehicle and the vehicle
        # in the target lane was too close, the vehicle will give up changing lanes and keep the original lane instead
        if self.target_lane != self.lane:
            target_lane = self.lane + 1 if self.target_lane > self.lane else self.lane - 1
            for i in range(len(vehicles)):
                temp_vehicle = vehicles[i]
                if temp_vehicle.id == self.id:
                    continue
                distance = Road.relation_distance(self.x, temp_vehicle.x)
                if temp_vehicle.lane == int(target_lane) and -5.0 < distance < 5.0:
                    self.keep_lane_force()

    def planning(self, vehicles):
        L_dv = Direction(1, 0)  # Longitudinal direction (the direction along the road)
        H_dv = Direction(-L_dv.y, L_dv.x)  # Lateral orientation

        dx = 0
        dy = 0

        self.graph_force()
        if self.target_lane == self.lane:
            self.keep_lane_force()
        else:
            self.change_lane_force()

        if ((self.state == 0 or self.state == 3) and self.neighborhoods[2] is not None
                and (self.neighborhoods[2].state == 0 or self.neighborhoods[2] == 3)):
            self.target_speed += 0.5 * (self.neighborhoods[2].target_speed - self.target_speed)

        self.new_L_force = self.target_speed
        self.new_L_force += self.new_G_force
        self.collision_avoid(vehicles)

        dx += TIMESTEP * self.new_L_force * L_dv.x + TIMESTEP * self.new_H_force * H_dv.x
        dy += TIMESTEP * self.new_L_force * L_dv.y + TIMESTEP * self.new_H_force * H_dv.y

        self.controller(self.x + dx, self.y + dy, ((dx ** 2 + dy ** 2) ** 0.5) / TIMESTEP)

        # desired_angle = math.atan2(dy, dx)
        # angle_error = desired_angle - self.heading
        #
        # self.heading += angle_error
        # self.new_x = self.x + dx
        # self.new_y = self.y + dy

        self.old_G_force = self.new_G_force
        self.old_L_force = self.new_L_force
        self.old_H_force = self.new_H_force
        self.last_target_speed = self.target_speed

    def controller(self, x2, y2, speed):
        self.speed = speed
        L1 = 3
        L2 = 10
        prefer_angle = math.atan2(y2 - self.y, x2 - self.x)
        angle_error = prefer_angle - self.heading
        lat_error = 0

        tan_sita = (-math.cos(angle_error) * lat_error - (L1 + L2) * math.sin(angle_error)) / (
                    L1 - (L1 + L2) * math.cos(angle_error) + math.sin(angle_error) * lat_error)

        k1 = 45.0 / math.atan(1.0)
        sita = math.atan(tan_sita)
        if (sita - self.old_sita) > (0.5 / k1):
            sita = self.old_sita + 0.5 / k1
        if sita > 25.0 / k1:
            sita = 25.0 / k1
        self.old_sita = sita
        tan_sita = math.tan(sita)

        temp_angle = tan_sita / L1 * speed * TIMESTEP
        self.heading += temp_angle

        temp_x = math.cos(self.heading) * speed * TIMESTEP
        self.new_x = self.x + temp_x

        temp_y = math.sin(self.heading) * speed * TIMESTEP
        self.new_y = self.y + temp_y

    def show_neighborhoods(self):
        print(self.id, end="：")
        for neighborhood in self.neighborhoods:
            if neighborhood is not None:
                print(neighborhood.id, end=",")
            else:
                print("None", end=",")
        print()
