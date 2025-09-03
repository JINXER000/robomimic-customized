"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import robomimic.envs.env_robosuite
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import robosuite


def extract_trajectory(
    env_meta,
    args, 
    initial_state, 
    states, 
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    done_mode = args.done_mode
    if env_meta['env_name'].startswith('PickPlace_'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    elif env_meta['env_name'].startswith('Libero_'):
        camera_names=['agentview', 'robot0_eye_in_hand']
    else: ## mimicgen, dexmimicgen
        camera_names=['frontview', 'birdview', 'agentview',  'robot0_eye_in_hand'] # sideview
    ## dexmimicgen
    if args.num_robots == 2:
        camera_names.append('robot1_eye_in_hand')
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # if t == 75:
        #     print("timestep 75")

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, args, initial_state, states, actions = x
    traj = extract_trajectory(
        env_meta=env_meta,
        args=args,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
    )
    return traj

def dataset_states_to_obs(args):
    num_workers = args.num_workers
    # Determine output path early and skip if it already exists to avoid creating rendering contexts
    input_file_name = os.path.basename(args.input)
    output_path = os.path.join(args.output_dir, input_file_name.replace(".hdf5", "_pc_instance.hdf5"))
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping...")
        return
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)



    if env_meta['env_name'].startswith('PickPlace_'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    elif env_meta['env_name'].startswith('Libero_'):
        camera_names=['agentview', 'robot0_eye_in_hand']
        # camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']

        from libero.libero import get_libero_path
        from libero.libero import benchmark
        # from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = args.dataset_type # can also choose libero_spatial, libero_object, etc.
        task_suite = benchmark_dict[task_suite_name]()

        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            task_name = task.name
            if task_name in args.input:
                task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                break

        env_meta['bddl_file'] = task_bddl_file
        env_meta['env_kwargs']['bddl_file_name'] = task_bddl_file
        env_meta['env_kwargs']["camera_segmentations"] = "instance"

        robots = env_meta['env_kwargs']['robots'][0]
        sides = ['right']
    else:
        # camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
        camera_names = env_meta['env_kwargs'].get('camera_names', ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_eye_in_hand'])

    ## dexmimicgen
    if args.num_robots == 2:
        env_meta['env_kwargs']["camera_segmentations"] = "instance"
        robots = env_meta['env_kwargs']['robots']
        sides = ["right", "left"]

    env_meta['env_kwargs']['output_all_pcds'] = args.output_all_pcds
        
    ## revise controller config for V1.5.1
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id >= 5:
        from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
        # Convert to composite controller config format
        updated_controller_configs = refactor_composite_controller_config(
            env_meta['env_kwargs']['controller_configs'], 
            robots,
            sides,
        )
        updated_controller_configs["type"] = "SWITCHABLE"

        env_meta['env_kwargs']['controller_configs'] = updated_controller_configs

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    # env = OffScreenRenderEnv(**env_meta['env_kwargs'])

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.input, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.input))
    print("output file: {}".format(output_path))

    # debug_ep_id = 20
    debug_ep_id = None

    total_samples = 0
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        for j in range(i, end):

            # debug
            if debug_ep_id is not None and debug_ep_id != j:
                continue

            ep = demos[j]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
            
        if debug_ep_id is not None:

            if len(states_list) > 0:
                trajs =worker([env_meta, args, initial_state_list[0], states_list[0], actions_list[0]])
            else:
                continue
        else:
            with multiprocessing.Pool(num_workers) as pool:
                trajs = pool.map(worker, [[env_meta, args, initial_state_list[j], states_list[j], actions_list[j]] for j in range(len(initial_state_list))]) 

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            ignore_keys = ['depth', 'segment']
            # ignore_keys = []
            for k in traj["obs"]:
                ignore = False
                for ignore_key in ignore_keys:
                    if ignore_key in k:
                        # skip keys that contain ignore_key
                        ignore = True
                        break
                if ignore:
                    continue

                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        
        del trajs

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='libero_spatial',
        help="Choose from libero_spatial, libero_object, DMG",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # dir of hdf5 to write 
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    ## if output all pcd
    parser.add_argument(
        "--output_all_pcds",
        type=bool, 
        default=False, 
        help="set to True only when train policy",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_robots",
        type=int,
        default=1,
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        type=bool,
        default=True,
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        type=bool,
        default=True,
        help="(optional) compress observations with gzip option in hdf5",
    )

    args = parser.parse_args()
    # If input is a directory, iterate over all .hdf5 files inside and process each
    if os.path.isdir(args.input):
        hdf5_files = [
            os.path.join(args.input, fname)
            for fname in sorted(os.listdir(args.input))
            if fname.lower().endswith(".hdf5") and not fname.endswith("_pc_instance.hdf5")
        ]
        if len(hdf5_files) == 0:
            print(f"No .hdf5 files found in directory: {args.input}")
        for dataset_path in hdf5_files:
            per_file_args = deepcopy(args)
            per_file_args.input = dataset_path
            print(f"\nProcessing dataset: {dataset_path}")
            dataset_states_to_obs(per_file_args)
    else:
        dataset_states_to_obs(args)
