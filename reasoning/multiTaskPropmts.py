import textwrap

from sumo.convoyVehicle import ConvoyVehicle


class MultiTaskPrompts:
    def __init__(self, ego: ConvoyVehicle):
        self.ego = ego
        self.state = ego.state
        self.delimiter = "####"  # separator

    def get_system_message(self):
        # avoid obstacles
        system_message_0 = textwrap.dedent(f"""\
                You are a large language model. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex highway driving scenarios.
                """)
        # join the convoy
        system_message_1 = textwrap.dedent(f"""\
                You are a large language model. Now you act as a mature driving assistant, who are driving one vehicle in a convoy of 8 vehicles.
                Your mission is to join the convoy.
                """)
        # leave the convoy
        system_message_2 = textwrap.dedent(f"""\
                You are a large language model. Now you act as a mature driving assistant, who are driving one vehicle in a convoy of 8 vehicles.
                Your mission is to drive the car out of the convoy, and you have to drive the vehicle at least {self.ego.communication_distance} meters away from the vehicles in the convoy.
                """)
        # switch to escort formation
        system_message_3 = textwrap.dedent(f"""\
                You are a large language model. Now you act as a mature driving assistant, who are driving one vehicle in a convoy of 8 vehicles.
                The convoy now needs to change from a normal formation to an escort formation.
                """)
        common_message = textwrap.dedent(f"""
                You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {self.delimiter}.

                Your response should use the following format:
                <Reasoning>
                <Reasoning>
                Response to user:{self.delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to IDLE, then output `1`>

                Make sure to include {self.delimiter} to separate every step.
                """)
        system_messages = [system_message_0, system_message_1,system_message_2, system_message_3]
        return system_messages[self.state] + common_message

    def get_human_message(self, scenario_description, available_actions):
        # avoid obstacles
        human_message_0 = textwrap.dedent(f"""\
                Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                Here is the current scenario:
                {self.delimiter} Driving scenario description:
                {scenario_description}
                {self.delimiter} Driving Intentions:
                0.If there is a convoy vehicle in front of the same lane, the IDLE will be output directly.
                1.Driving carefully and void collision.
                2.The safe distance for changing lanes is at least 15 meters.
                3.Keep a safe distance of at least 15 meters from the ahead of the surrounding vehicles.
                4.Try to maintain the desired speed of 25 m/s when there is no vehicle ahead in the same lane.
                5.Changing lanes does not require consideration of collision with the convoy vehicles.
                6.Limit speed is 15~30 m/s.
                {self.delimiter} Available actions:
                {available_actions}
                You can stop reasoning once you have a valid action to take.
        
                """)
        # join the convoy
        human_message_1 = textwrap.dedent(f"""\
                Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                Here is the current scenario:
                {self.delimiter} convoy scenario description:
                {scenario_description}
                {self.delimiter} Driving Intentions:
                1. Your should change lane to the middle lane.
                2. If you are behind the convoy of vehicles, you need to speed up to catch up with the convoy.
                3. Limit speed is 15~30 m/s.
                {self.delimiter} Available actions:
                {available_actions}
        s
                You can stop reasoning once you have a valid action to take. 
                """)
        # leave the convoy
        human_message_2 = textwrap.dedent(f"""\
                Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                Here is the current scenario:
                {self.delimiter} convoy scenario description:
                {scenario_description}
                {self.delimiter} Driving Intentions:
                1. If there are no other convoy vehicles in front or behind the same lane, you can speed up or slow down away from the convoy.
                2. Limit speed is 15~30 m/s.
                3. Changing lanes does not require consideration of collision with the convoy vehicles.
                {self.delimiter} Available actions:
                {available_actions}
                You can stop reasoning once you have a valid action to take. 
                """)
        # switch to escort formation
        related_position_description = {
            "veh1":"You should behind the veh4 about 10 meters on the same lane",
            "veh2":"You should behind the veh4 about 5 meters on the right lane",
            "veh3": "You should behind the veh4 about 5 meters on the left lane",
            "veh4": "You are the protected vehicle.",
            "veh5": "You should ahead the veh4 about 5 meters on the left lane",
            "veh6": "You should ahead the veh4 about 5 meters on the right lane",
            "veh7": "You should ahead the veh4 about 10 meters on the same lane",
            "veh8": "You should ahead the veh7 about 10 meters on the same lane",
        }
        lanes = {0: "the rightmost lane", 1: "the middle lane", 2: "the leftmost lane"}
        human_message_3 = textwrap.dedent(f"""\
                Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                Here is the current scenario:
                {self.delimiter} convoy scenario description:
                {scenario_description}
                {self.delimiter} Driving Intentions:
                1. The current vehicle to be protected is veh4.
                2. Your desired lane in the escort formation is {lanes[self.ego.desired_lane]}.
                3. {related_position_description[self.ego.id]}
                4. Your desired speed is 25 m/s.
                {self.delimiter} Available actions:
                {available_actions}
                You can stop reasoning once you have a valid action to take. 
                """)
        human_messages = [human_message_0, human_message_1,human_message_2, human_message_3]
        return human_messages[self.state].replace("        ", "")
