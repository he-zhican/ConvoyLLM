import os
import textwrap
from rich import print
from typing import List

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler

from reasoning.multiTaskPropmts import MultiTaskPrompts
from sumo.convoyVehicle import ConvoyVehicle

delimiter = "####"
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
         You are driving on a road with 3 lanes, and you are currently driving in the second lane from the left. Your current position is `(453.30, 5.40)`, speed is 25.00 m/s, and lane position is 453.30 m.
        There are other all vehicles driving around you, and below is their basic information,
        - Vehicle `flow_1.5` is driving on the lane to your left and is behind of you. 
        The position of it is `(426.61, 9.00)`, speed is 22.00 m/s, acceleration is 0.00 m/s^2, and lane position is 426.61 m.
        - Vehicle `flow_1.3` is driving on the lane to your right and is ahead of you. 
        The position of it is `(484.14, 1.80)`, speed is 20.00 m/s, acceleration is 0.00m/s^2, and lane position is 484.14 m.
        - Vehicle `flow_1.14` is driving on the same lane as you and is ahead of you. 
        The position of it is `(500.69, 5.40)`, speed is 21.49 m/s, acceleration is -0.06 m/s^2, and lane position is 500.69 m.


        {delimiter} Your available actions:
        Turn-left - change lane to the left of the current lane Action_id: 0
        IDLE - remain in the current lane with current speed and cancel changing lanes Action_id: 1
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\

        """)


class DecisionMaker:
    def __init__(self, temperature: float = 0, verbose: bool = False) -> None:
        self.llm = ChatOpenAI(
            temperature=temperature,
            callbacks=[
                OpenAICallbackHandler()
            ],
            model=os.getenv("OPENAI_CHAT_MODEL"),
            max_tokens=2000,
            request_timeout=60,
            streaming=True,
        )

    def few_shot_decision(self, ego: ConvoyVehicle = None, scenario_description: str = "Not available",
                          available_actions: str = "Not available",
                          fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        multiTaskPropmts = MultiTaskPrompts(ego)
        system_message = multiTaskPropmts.get_system_message()
        human_message = multiTaskPropmts.get_human_message(scenario_description, available_actions)

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store
