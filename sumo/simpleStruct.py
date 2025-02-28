TIMESTEP = 0.025


class Direction:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Road:
    road_id = "E0"
    direction = "forward"
    lane_type = "straight"
    lane_num = 3
    lane_width = 3.6
    speed_limit = 35

    @staticmethod
    def relation_distance(x1, x2):
        return x2 - x1

    # Get the distance between the point (x,y) and the left and right lane lines, left distance:
    # the distance from the left lane line; right distance: distance from the right lane line
    @staticmethod
    def get_lanes_distance(y, lane):
        left_distance = (lane + 1) * Road.lane_width - y
        right_distance = Road.lane_width - left_distance
        return left_distance, right_distance
