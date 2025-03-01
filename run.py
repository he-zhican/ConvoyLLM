import random

import yaml
import os
from rich import print

from reasoning.scenarioDescription import ScenarioDescription
from reasoning.decisionMaker import DecisionMaker
from sharedMemory.sharedMemory import SharedMemory

from sumo.mutilEnv import MutilEnv

test_list_seed = [52, 5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 5326,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31, 8317,
                  123, 456, 789, 101, 202, 303, 404, 505, 606, 707,
                  808, 909, 111, 222, 333, 444, 555, 666, 777, 888,
                  999, 121, 232, 343, 454, 565, 676, 787, 898, 919]


def setup_env(config):
    os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
    os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    os.environ["OPENAI_EMBEDDING_MODEL"] = config['OPENAI_EMBEDDING_MODEL']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    setup_env(config)

    sumo_home = config["SUMO_HOME"]
    task = config["task"]
    all_vehicles_decision = config["all_vehicles_decision"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"] + f"/task_{task}"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {}\n".format(
            memory_path, result_folder, few_shot_num))

    # agent_memory = DrivingMemory(db_path=memory_path)
    memory_task_0 = SharedMemory(db_path=memory_path)
    memory_task_1 = SharedMemory(db_path=memory_path + "_task_1")
    memory_task_2 = SharedMemory(db_path=memory_path + "_task_2")
    memory_task_3 = SharedMemory(db_path=memory_path + "_task_3")
    # memory_task_0.combineMemory(agent_memory)
    memory_list = [memory_task_0, memory_task_1, memory_task_2, memory_task_3]

    episode = 0
    env = MutilEnv("human", sumo_home=sumo_home, task=task, result_folder=result_folder)
    while episode < config["episodes_num"]:
        success = False
        envType = f'sumo-highway-multi-{task}'
        result_prefix = f"sumo_highway_multi_{episode}"
        # seed = random.choice(test_list_seed)
        seed = test_list_seed[episode]
        obses, _ = env.reset(seed=seed)
        env.initialize_video_writer(result_folder + "/" + result_prefix + ".mp4")
        # scenario and driver agent setting
        database_path = result_folder + "/" + result_prefix + ".db"

        # Create a ScenarioDescription class for each vehicle in the convoy and record {"vhe 1":sce} with a dictionary
        sce_dict = {}
        for i, cv in enumerate(env.convoy_vehicles):
            sce = ScenarioDescription(env, cv)
            sce_dict[env.cv_names[i]] = sce

        decisionMaker = DecisionMaker(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                if not all_vehicles_decision:
                    all_lead_cvs = env.get_all_lead_cvs()
                action_list = []
                for cv in env.convoy_vehicles:
                    if not all_vehicles_decision:
                        if cv not in all_lead_cvs:
                            action_list.append(1)
                            continue
                    print(f"[cyan]The {cv.id} retreive similar memories...[/cyan]")
                    fewshot_messages = []
                    fewshot_answers = []
                    fewshot_actions = []
                    fewshot_results = memory_list[cv.state].retriveMemory(
                        sce_dict[cv.id], few_shot_num) if few_shot_num > 0 else []

                    for fewshot_result in fewshot_results:
                        fewshot_messages.append(
                            fewshot_result["human_question"])
                        fewshot_answers.append(fewshot_result["LLM_response"])
                        fewshot_actions.append(fewshot_result["action"])
                        mode_action = max(set(fewshot_actions), key=fewshot_actions.count)
                        mode_action_count = fewshot_actions.count(mode_action)
                    if few_shot_num == 0:
                        print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                    else:
                        print("[green4]Successfully find[/green4]", len(
                            fewshot_actions), "[green4]similar memories![/green4]")

                    sce_descrip = sce_dict[cv.id].describe()
                    avail_action = sce_dict[cv.id].availableActionsDescription()
                    print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                    # print('[cyan]Available actions: [/cyan]\n',avail_action)
                    action, response, human_question, fewshot_answer = decisionMaker.few_shot_decision(
                        ego=cv, scenario_description=sce_descrip, available_actions=avail_action,
                        fewshot_messages=fewshot_messages, fewshot_answers=fewshot_answers,
                    )
                    docs.append({
                        "sce_descrip": sce_descrip,
                        "human_question": human_question,
                        "response": response,
                        "action": action,
                        "sce": sce_dict[cv.id],
                        "state": cv.state
                    })
                    action_list.append(action)
                    already_decision_steps += 1

                obses, rewards, dones, truncateds, _ = env.step(action_list)

                print("=================================")

                if True in dones:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break
                if True in truncateds:
                    success = True
                    print("[green]Simulation finnish successfully! [/green] ")
                    break
        finally:
            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | success: {} | aver_speeds: {} | File prefix: {} \n"
                    .format(episode, seed, already_decision_steps, success, env.get_aver_speeds(), result_prefix))

                print("[yellow]Do you want to add[/yellow]", len(docs) // 5,
                      "[yellow]new memory item to update memory module?[/yellow]", end="")
                choice = input("(Y/N): ").strip().upper()
                if choice == 'Y':
                    cnt = 0
                    for i in range(0, len(docs)):
                        if i % 5 == 1:
                            memory_list[docs[i]["state"]].addMemory(
                                docs[i]["sce_descrip"],
                                docs[i]["human_question"],
                                docs[i]["response"],
                                docs[i]["action"],
                                docs[i]["sce"],
                                comments="no-mistake-direct"
                            )
                            cnt += 1
                    print("[green] Successfully add[/green] ", cnt,
                          " [green]new memory item to update memory module.[/green]. "
                          "Now the memory_task_0 database has ", len(
                            memory_list[0].scenario_memory._collection.get(include=['embeddings'])['embeddings']),
                          " items. memory_task_1 database has ", len(
                            memory_list[1].scenario_memory._collection.get(include=['embeddings'])['embeddings']),
                          "memory_task_2 database has", len(
                            memory_list[2].scenario_memory._collection.get(include=['embeddings'])['embeddings']))
                else:
                    print("[blue]Ignore these new memory items[/blue]")

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close_video_writer()
    env.close()
