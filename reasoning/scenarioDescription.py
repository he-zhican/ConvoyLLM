from sumo.mutilEnv import MutilEnv
from sumo.convoyVehicle import ConvoyVehicle
from sumo.vehicle import Vehicle
from sumo.simpleStruct import Road

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed, and cancel changing lane action.',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class ScenarioDescription:
    def __init__(self, env: MutilEnv, ego: ConvoyVehicle) -> None:
        self.env = env
        self.ego = ego

    def getSurroundVehicles(self, vehicles_count: int):
        return self.ego.find_surround_evs(self.env.get_all_evs(), vehicles_count)

    def availableActionsDescription(self) -> str:
        availableActionDescription = 'Your available actions are: \n'
        availableActions = range(self.env.action_space.n)
        for action in availableActions:
            availableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + '\n'
        return availableActionDescription

    def describeConvoyVehicle(self) -> str:
        numLanes = Road.lane_num
        if numLanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:
            egoLaneRank = self.ego.lane
            lane_description = f"You are driving on a road with {numLanes} lanes, "
            if egoLaneRank == 2:
                description = lane_description + f"and you are currently driving in the leftmost lane."
            elif egoLaneRank == 0:
                description = lane_description + f"and you are currently driving in the rightmost lane."
            else:
                description = lane_description + f"and you are currently driving in the middle lane."

        description += f"Your current position is `({self.ego.x:.2f}, {self.ego.y:.2f})`, speed is {self.ego.last_target_speed:.2f} m/s, and lane position is {self.ego.x:.2f} m."

        if self.ego.target_lane - self.ego.lane == 1:
            description += " and you are changing lanes to the left.\n"
        elif self.ego.target_lane - self.ego.lane == -1:
            description += " and you are changing lanes to the right.\n"
        else:
            description += "\n"

        return description

    def getSVRelativeState(self, sv: Vehicle) -> str:
        relativePosition = sv.x - self.ego.x
        if relativePosition >= 0:
            return 'is ahead of you'
        else:
            return 'is behind of you'

    # Get the closest environment vehicles in each lane to the front and back of ego vehicle
    def processSurroundEVs(self, e_vehicles):
        # [lane0_front,lane0_behind,lane1_front,lane1_behind,lane2_front,lane2_behind]
        surround_evs = [None] * 6
        for ev in e_vehicles:
            if ev is not None:
                ego_x = self.ego.x
                dis = Road.relation_distance(ego_x, ev.x)
                if ev.lane == 0:
                    if dis >= 0 and (surround_evs[0] is None or dis < Road.relation_distance(ego_x, surround_evs[0].x)):
                        surround_evs[0] = ev
                    elif dis < 0 and (surround_evs[1] is None or dis > Road.relation_distance(ego_x, surround_evs[1].x)):
                        surround_evs[1] = ev
                elif ev.lane == 1:
                    if dis >= 0 and (surround_evs[2] is None or dis < Road.relation_distance(ego_x, surround_evs[2].x)):
                        surround_evs[2] = ev
                    elif dis < 0 and (surround_evs[3] is None or dis > Road.relation_distance(ego_x, surround_evs[3].x)):
                        surround_evs[3] = ev
                elif ev.lane == 2:
                    if dis >= 0 and (surround_evs[4] is None or dis < Road.relation_distance(ego_x, surround_evs[4].x)):
                        surround_evs[4] = ev
                    elif dis < 0 and (surround_evs[5] is None or dis > Road.relation_distance(ego_x, surround_evs[5].x)):
                        surround_evs[5] = ev

        return surround_evs

    def describeSurroundEVs(self) -> str:
        e_vehicles = self.getSurroundVehicles(10)
        surround_evs = self.processSurroundEVs(e_vehicles)
        lanes = {0: "the rightmost lane", 1: "the middle lane", 2: "the leftmost lane"}
        if all(sv is None for sv in surround_evs):
            SVDescription = f'There are no other environment vehicles driving near you, so you should change lane to your desired lane that is {lanes[self.ego.desired_lane]}\n'
            return SVDescription
        else:
            SVDescription = 'There are other environment vehicles driving around you, and below is their basic information:\n'
            for i, sv in enumerate(surround_evs):
                if sv is not None:
                    SVDescription += f"- vehicle '{sv.id}' is diving in {lanes[sv.lane]}, and {self.getSVRelativeState(sv)}."
                    SVDescription += f"The speed of it is {sv.speed:.2f} m/s, acceleration is {sv.acceleration:.2f} m/s^2, and lane position is {sv.x:.2f} m.\n"
                else:
                    SVDescription += f"- There are no vehicle driving in {lanes[i//2]} which is {'ahead' if i%2==0 else 'behind'} of you.\n"

            return SVDescription

    def describeNeighborhoods(self) -> str:
        neighborhoods = self.ego.neighborhoods
        related_lanes = ["right", "same", "left"]
        if all(neighbor is None for neighbor in neighborhoods):
            NVDescription = f'There are no other convoy vehicles driving near you.\n'
            return NVDescription
        else:
            NVDescription = 'There are other convoy vehicles driving around you, and below is their basic information:\n'
            for i, neighbor in enumerate(neighborhoods):
                if neighbor is not None:
                    NVDescription += f"- The neighbor convoy vehicle '{neighbor.id}' is driving on the {related_lanes[i//2]} lane to you, and {self.getSVRelativeState(neighbor)}."
                    NVDescription += f"The speed of it is {neighbor.last_target_speed:.2f} m/s, and lane position is {neighbor.x:.2f} m.\n"
            return NVDescription

    def describe(self,) -> str:
        CVDescription = self.describeConvoyVehicle()
        SVDescription = self.describeSurroundEVs()
        NVDescription = self.describeNeighborhoods()

        description = CVDescription
        if self.ego.state == 0:
            description += SVDescription + NVDescription
        else:
            description += NVDescription

        return description
