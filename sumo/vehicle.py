import traci
import math


class Vehicle:
    def __init__(self, vehicle_id):
        self.id = vehicle_id
        self.lane = -1  # Lane index: 0,1,2 --> from right to left
        self.x = 0
        self.y = 0
        self.heading = 0  # (0,360)
        self.speed = 0  # m/s
        self.length = 4.0
        self.width = 1.8
        self.acceleration = 0
        self.steering = 0
        self.max_dec = 2  # m/s^2
        self.max_acc = 2.0  # m/s^2

    def update_state(self, client: traci):
        # Get information such as vehicle coordinates,lane,speed
        self.x, self.y = client.vehicle.getPosition(self.id)
        self.lane = client.vehicle.getLaneIndex(self.id)
        self.length = client.vehicle.getLength(self.id)
        self.width = client.vehicle.getWidth(self.id)
        self.acceleration = client.vehicle.getAcceleration(self.id)
        temp_speed = client.vehicle.getSpeed(self.id)
        if temp_speed != 0:
            self.speed = temp_speed
        self.heading = client.vehicle.getAngle(self.id)
        self.heading = (90 - self.heading) / 180 * math.pi

    def show_state(self):
        print("id:", self.id)
        print(f"(x,y):({self.x},{self.y})")
        print(f"lane:{self.lane}")
        print(f"speed:{self.speed}")
        print(f"heading:{self.heading}")
        print(f"width:{self.width}")
        print(f"length:{self.length}")
