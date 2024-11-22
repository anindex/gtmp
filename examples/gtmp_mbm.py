from typing import Dict, List, Union
import jax
from jax import jit, vmap, random
import hydra
import omegaconf
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from chrono import Timer
from gtmp.files import get_configs_path, get_data_path
from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan, interpolate_path
from gtmp.objectives.sphere_approximation import CollisionSphere
from gtmp.objectives.primitives import SphereField, CylinderField, CuboidField
from gtmp.objectives.costs import CostInfinite, CostCollision
from gtmp.kinematics.robot import Robot
from gtmp.metrics import compute_metrics
from gtmp.pybullet import PyBulletSimulator

from kinax.model import FRANKA_PANDA

ROBOT_GRIPPER_DIMS = {
    'panda': 2,
}

COLL_DICT = {
    'panda': {
        'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4,
        'panda_link5': 5, 'panda_link6': 6, 'panda_link7': 7, 'panda_hand': 8
    },
}

ROBOT_JOINTS = {
    "panda": [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        ],
}

def problem_dict_to_rax(
        robot: str,
        problem: Dict[str, List[Dict[str, Union[float, List[float]]]]],
        link_dict: Dict[str, int],
        ignore_names: List[str] = []
    ) -> CostInfinite:
    robot_model = Robot.create(
        model_path=FRANKA_PANDA
    )
    dim = len(robot_model.system.joint_ids) - ROBOT_GRIPPER_DIMS[robot]

    sdfs = []
    if len(problem["sphere"]) > 0:
        pos = [obj["position"] for obj in problem["sphere"] if obj['name'] not in ignore_names]
        rad = [obj["radius"] for obj in problem["sphere"] if obj['name'] not in ignore_names]
        sphere_field = SphereField(
            centers=jnp.array(pos),
            radii=rad
        )
        sdfs.append(sphere_field)

    if len(problem["cylinder"]) > 0:
        pos = [obj["position"] for obj in problem["cylinder"] if obj['name'] not in ignore_names]
        eulers = [obj["orientation_euler_xyz"] for obj in problem["cylinder"] if obj['name'] not in ignore_names]
        rad = [obj["radius"] for obj in problem["cylinder"] if obj['name'] not in ignore_names]
        length = [obj["length"] for obj in problem["cylinder"] if obj['name'] not in ignore_names]

        cyl_field = CylinderField.from_eulers(
            centers=jnp.array(pos),
            eulers=jnp.array(eulers),
            radii=jnp.array(rad),
            lengths=jnp.array(length)
        )
        sdfs.append(cyl_field)

    if len(problem["box"]) > 0:
        pos = [obj["position"] for obj in problem["box"] if obj['name'] not in ignore_names]
        eulers = [obj["orientation_euler_xyz"] for obj in problem["box"] if obj['name'] not in ignore_names]
        half_ext = [obj["half_extents"] for obj in problem["box"] if obj['name'] not in ignore_names]

        cub_field = CuboidField.from_eulers(
            centers=jnp.array(pos),
            eulers=jnp.array(eulers),
            half_extents=jnp.array(half_ext)
        )
        sdfs.append(cub_field)
    coll_field = CollisionSphere.create(sdf_list=sdfs, robot_name=robot, link_dict=link_dict)
    obst_cost = CostCollision.create(
        dim=dim,
        field=coll_field
    )
    cost_fn = CostInfinite(
        dim=dim,
        buffer_dim=ROBOT_GRIPPER_DIMS[robot],  # gripper dims
        traj_len=1,
        cost_list=(obst_cost,),
        robot=robot_model
    )

    return cost_fn, robot_model, dim


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_mbm")
def main(cfg: omegaconf.DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    robot = cfg.robot
    problem = cfg.problem
    index = cfg.index

    data_dir = get_data_path() / f"{robot}"
    with open(data_dir / "problems.pkl", 'rb') as f:
        data = pickle.load(f)

    if not problem:
        problem = list(data['problems'].keys())[0]

    if problem not in data['problems']:
        raise RuntimeError(
            f"""No problem with name {problem}!
                Existing problems: {list(data['problems'].keys())}"""
            )

    problems = data['problems'][problem]
    try:
        problem_data = next(problem for problem in problems if problem['index'] == index)
    except StopIteration:
        raise RuntimeError(f"No problem in {problem} with index {index}!")

    start = jnp.array(problem_data['start'])
    goals = jnp.array(problem_data['goals'])
    valid = problem_data['valid']
    
    coll_dict = COLL_DICT[robot]
    cost_fn, robot_model, dim = problem_dict_to_rax(robot, problem_data, coll_dict)

    # planner
    robot_model = Robot.create(
        model_path=FRANKA_PANDA
    )
    dim = len(robot_model.system.joint_ids) - ROBOT_GRIPPER_DIMS[robot]
    coll_dict = COLL_DICT[robot]
    if cfg.planner.name == 'straight':
        gtmp = jit(vmap(gtmp_plan, in_axes=(0, None)))
        num_interpolate = 100
    elif cfg.planner.name == 'akima':
        gtmp = jit(vmap(gtmp_akima_plan, in_axes=(0, None)))
        num_interpolate = 10
    gtmp_state = GTMPState.create(
        dim=dim,
        q=start,
        goals=goals,
        bounds=robot_model.q_limits[:dim],
        occ_map=cost_fn,
        **cfg.planner.params
    )

    if valid:
        # plan
        keys = jax.random.split(rng_key, cfg.num_plans)
        start = time.time()
        paths = gtmp(keys, gtmp_state)
        print(f"JIT Time taken: {time.time() - start} seconds")
        with Timer() as timer:    
            paths = gtmp(keys, gtmp_state)
        metrics = compute_metrics(paths)
        print(f"Time taken: {timer.elapsed} seconds")
        print(f"Collision-free percentage: {metrics[0]}")
        print(f"Averaged path length: {metrics[1]}")
        print(f"Path Diversities: {metrics[2]}")
        print(f"Min Cosin: {metrics[3]}")
        print(f"Mean Cosin: {metrics[4]}")
        in_paths = vmap(interpolate_path, in_axes=(0, None))(paths.path[~paths.collision], num_interpolate)
        in_paths = in_paths.reshape(in_paths.shape[0], -1, in_paths.shape[-1])
        colls = cost_fn.get_collisions(in_paths).astype(jnp.bool).any(-1)
        collision_free = 1 - colls.mean()
        print(f"Collision-free percentage after interpolating: {collision_free}")
        solved = collision_free > 0.
    else:
        print("Problem is invalid!")
        solved = False

    if valid and not solved:
        print("Failed to solve problem! Displaying start and goals.")
        print(start)
        for goal in goals:
            print(goal)

    free_paths = np.array(in_paths[~colls])
    sim = PyBulletSimulator(str(data_dir / f"{robot}_spherized.urdf"), ROBOT_JOINTS[robot])
    sim.add_environment_from_problem_dict(problem_data, False)
    sim.animate_plans(free_paths)


if __name__ == "__main__":
    main()
