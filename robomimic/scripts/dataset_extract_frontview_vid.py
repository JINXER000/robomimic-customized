
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
import imageio
import robosuite

def extract_vid(
    env_meta,
    args, 
    initial_state, 
    states, 
    actions,
    vid_path,
):

    done_mode = args.done_mode

    camera_names = ['agentview']
 
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

    video_writer = imageio.get_writer(vid_path, fps=20)
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # TODO: record video
        img_key = 'agentview_image' if 'agentview_image' in obs else 'agentview_rgb'
        front_img = obs[img_key]
        video_writer.append_data(front_img)

        # update for next iter
        obs = deepcopy(next_obs)

    video_writer.close()

    return video_writer

def worker(x):
    env_meta, args, initial_state, states, actions, output_path = x

    if os.path.exists(output_path):
        print(f"Output dir {output_path} already exists. Skipping...")
        return 
    
    video_writer = extract_vid(
        env_meta=env_meta,
        args=args,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
        vid_path=output_path,
    )
    return 

def dataset_to_vids(args):
    num_workers = args.num_workers
    # Determine output path early and skip if it already exists to avoid creating rendering contexts
    input_file_name = os.path.basename(args.input)
    output_dir = os.path.join(args.output_dir, input_file_name.replace(".hdf5", "vids"))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)


    if env_meta['env_name'].startswith('Libero_'):

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

        robots = env_meta['env_kwargs']['robots'][0]
        sides = ['right']


    ## dexmimicgen
    if args.num_robots == 2:
        robots = env_meta['env_kwargs']['robots']
        sides = ["right", "left"]

    camera_names=['agentview']
        
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

    print("input file: {}".format(args.input))

    # debug_ep_id = 0
    debug_ep_id = None

    total_samples = 0
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        output_path_list = []
        for j in range(i, end):

            # debug
            if debug_ep_id is not None and debug_ep_id != j:
                continue

            ep = demos[j]

            output_path = os.path.join(output_dir, ep + ".mp4")
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
            output_path_list.append(output_path)
        if debug_ep_id is not None:

            if len(states_list) > 0:
                trajs =worker([env_meta, args, initial_state_list[0], states_list[0], actions_list[0], output_path_list[0]])
            else:
                continue
        else:
            with multiprocessing.Pool(num_workers) as pool:
                trajs = pool.map(worker, [[env_meta, args, initial_state_list[j], states_list[j], actions_list[j], output_path_list[j]] for j in range(len(initial_state_list))]) 

    f.close()



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
            dataset_to_vids(per_file_args)
    else:
        dataset_to_vids(args)
