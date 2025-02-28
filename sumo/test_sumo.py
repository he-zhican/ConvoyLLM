import random

import math

import traci
from traci.exceptions import TraCIException
from convoyVehicle import ConvoyVehicle
from simpleStruct import Road, Direction
from vehicle import Vehicle

# 假设已经定义了 Direction, ControlCar, Vehicle, DecisionMaker 类和其他相关变量
SIMULATION_DURATION = 72  # 仿真总时长，毫秒

all_cvs = []  # 所有编队车辆
all_evs = []  # 所有环境车辆


def start_sumo():
    try:
        # 定义 SUMO 可执行文件路径
        sumo_binary = "sumo-gui"  # 如果你想用GUI模式，使用 "sumo-gui"，否则使用 "sumo"
        # 定义网络和配置文件
        sumo_config = "../configs/convoy_0.sumocfg"  # SUMO 配置文件，包含地图和仿真参数
        # 启动 SUMO 模拟
        traci.start([sumo_binary, "-c", sumo_config, "--random", "--lanechange.duration", "1.5"])
    except TraCIException:
        input("无法连接到SUMO服务器，请检查配置并按回车键重试。")
        return


def main():
    # 所有受控车辆的id
    cv_names = ["veh1", "veh2", "veh3", "veh4", "veh5", "veh6", "veh7", "veh8"]
    desired_lanes = [1, 0, 1, 0, 1, 0, 1, 0]
    road = Road()

    for i, cv_id in enumerate(cv_names):
        temp_v = ConvoyVehicle(cv_id, desired_lanes[i])
        all_cvs.append(temp_v)

    all_cvs[0].state = 1

    count = 0

    start_sumo()
    simulation_time = 0
    while simulation_time < SIMULATION_DURATION:
        count += 1
        traci.simulationStep()
        simulation_time = traci.simulation.getTime()
        traci.gui.trackVehicle("View #0", "veh1")

        # if simulation_time % 10 == 0:
        #     spawn_random_vehicle_around(traci, all_cvs[7], count)

        vehicles_list = traci.vehicle.getIDList()
        for c in vehicles_list:
            if c not in cv_names:
                temp_v = Vehicle(c)
                temp_v.update_state(traci)
                print(f"{temp_v.id}'s speed is {temp_v.speed}")
                all_evs.append(temp_v)

        for c in all_cvs:
            c.update_state(traci)
            # c.show_state()

        for c in all_cvs:
            c.find_surround_evs(all_evs)
            c.find_neighborhoods(all_cvs)
            # c.show_neighborhoods()
            c.void_obstacles(all_evs)
            c.planning(all_cvs)
        for c in all_cvs:
            traci.vehicle.moveToXY(c.id, "", c.lane, c.new_x, c.new_y, angle=90 - c.heading * 180 / math.pi, keepRoute=2)

        all_evs.clear()

    traci.close()


if __name__ == "__main__":
    main()
