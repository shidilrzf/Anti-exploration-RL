from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu
import argparse
import torch
import uuid
from rlkit.core import logger
from rlkit.core.eval_util import get_generic_path_information
import pathlib

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['trainer/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        ptu.set_gpu_mode(True, gpu_id=args.device_id)
        policy.to(ptu.device)
    paths = []
    for i in range(10):
        path = rollout(
            env,
            policy,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        paths.append(path)

    for k, v in get_generic_path_information(paths).items():
        logger.record_tabular(k, v)
        print('{} : {}'.format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--device-id', type=int, default=6, help='GPU device id (default: 6')
    args = parser.parse_args()
    args.file = pathlib.Path(__file__).parent / '../examples/logs/{}/params.pkl'.format(args.file)

    torch.manual_seed(args.seed)
    simulate_policy(args)
