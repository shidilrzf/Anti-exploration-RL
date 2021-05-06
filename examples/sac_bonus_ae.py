from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.sac_ae import SAC_AETrainer

from rlkit.torch.networks import FlattenMlp, VAE
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


import argparse

import gym
import d4rl
import numpy as np
import torch
import time


def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    # q and policy netwroks
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)

    # initialize with bc or not
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    ).to(ptu.device)

    # if bonus: define bonus networks
    if not variant['offline']:
        checkpoint = torch.load(variant['bonus_path'], map_location=ptu.device)
        if variant['BC']:
            bonus_network = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=[M, M],
                std=variant['std']).to(ptu.device)
            bonus_network.load_state_dict(checkpoint['policy_state_dict'])
        else:
            bonus_network = VAE(
                input_sizes=[obs_dim, action_dim],
                latent_size=args.latent_size,
            ).to(ptu.device)
            bonus_network.load_state_dict(checkpoint['network_state_dict'])

        print('Loading bonus model: {}'.format(variant['bonus_path']))

    if variant['deterministic']:
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy)
    else:
        eval_path_collector = CustomMDPPathCollector(
            eval_env,
        )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    dataset = d4rl.qlearning_dataset(eval_env)
    load_hdf5(dataset, replay_buffer)

    if variant['normalize']:
        obs_mu, obs_std = dataset['observations'].mean(axis=0), dataset['observations'].std(axis=0)
        bonus_norm_param = [obs_mu, obs_std]
    else:
        bonus_norm_param = [None] * 2

    # shift the reward
    if variant['reward_shift'] is not None:
        rewards_shift_param = min(dataset['rewards']) - variant['reward_shift']
        print('.... reward is shifted : {} '.format(rewards_shift_param))
    else:
        rewards_shift_param = None

    reward_scale = 1 / (max(dataset['rewards']) - min(dataset['rewards']))

    if variant['offline']:
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            rewards_shift_param=rewards_shift_param,
            **variant['trainer_kwargs']
        )
        print('Agent of type offline SAC created')

    elif variant['bonus'] == 'bonus_add':
        trainer = SAC_AETrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bonus_network=bonus_network,
            bc_bonus=variant['BC'],
            beta=variant['bonus_beta'],
            use_bonus_critic=variant['use_bonus_critic'],
            use_bonus_policy=variant['use_bonus_policy'],
            use_log=variant['use_log'],
            norm_param=bonus_norm_param,
            rewards_shift_param=rewards_shift_param,
            device=ptu.device,
            reward_scale=reward_scale,
            **variant['trainer_kwargs']
        )
        print('Agent of type SAC + VAE additive bonus created')
    else:
        raise ValueError('Not implemented error')

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sac_bonus')
    parser.add_argument("--env", type=str, default='walker2d-medium-v0')
    parser.add_argument('--num_epochs', default=2000, type=int)

    # sac
    parser.add_argument('--qf_lr', default=1e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_lr', default=1e-5, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--no_automatic_entropy_tuning', action='store_true', default=False, help='no automatic entropy tuning')

    # bonus
    parser.add_argument('--offline', action='store_true', default=False, help='offline sac')
    parser.add_argument('--bonus', type=str, default='bonus_add', help='different type of bonus: bonus_add, bonus_mlt')  # Q + bonus or Q * bonus
    parser.add_argument('--beta', default=1, type=float, help='beta for the bonus')

    parser.add_argument("--root_path", type=str, default='/home/shideh/', help='path to the bonus model')
    parser.add_argument("--bonus_model", type=str, default=None, help='name of the bonus model')
    parser.add_argument('--bonus_type', type=str, default='actor-critic', help='use bonus in actor, critic or both')
    parser.add_argument('--BC', action='store_true', default=False, help='BC as bonus')
    parser.add_argument('--std', type=float, default=None, help='std of BC network')
    parser.add_argument('--latent_size', default=12, type=int)
    parser.add_argument('--normalize', action='store_true', default=False, help='use normalization in bonus')
    parser.add_argument('--reward_shift', default=None, type=int, help='minimum reward')
    parser.add_argument('--initialize_Q', action='store_true', default=False, help='initialize Q with bonus')
    parser.add_argument('--use_log', action='store_true', default=False, help='use log(bonus(s, a) otherwise 1 - bonus(s, a)')

    # initialize with bc
    parser.add_argument("--bc_model", type=str, default=None, help='name of pretrained bc model')

    # d4rl
    parser.add_argument('--dataset_path', type=str, default=None, help='d4rl dataset path')

    # evaluation
    parser.add_argument('--deterministic', action='store_true', default=False, help='determinstic policy for evaluation')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device-id', type=int, default=0, help='GPU device id (default: 0')
    args = parser.parse_args()

    # noinspection
    bonus_path = 'models/{}'.format(args.bonus_model)
    print(bonus_path)
    variant = dict(
        algorithm="SAC",
        version="normal",
        # initialize with bc
        bc_model=None,
        # bonus
        offline=args.offline,
        bonus=args.bonus,
        bonus_path=bonus_path,
        bonus_beta=args.beta,
        BC=args.BC,
        std=args.std,
        use_log=args.use_log,
        replay_buffer_size=int(1E6),
        layer_size=256,
        buffer_filename=args.env,  # halfcheetah_101000.pkl',
        load_buffer=True,
        env_name=args.env,
        seed=args.seed,
        # bonus_type
        use_bonus_policy=False,
        use_bonus_critic=False,
        initialize_Q=args.initialize_Q,
        # use normalization for bonus
        normalize=args.normalize,

        # make reward positive
        reward_shift=args.reward_shift,

        # evaluation
        deterministic=args.deterministic,

        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples,

        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            alpha_lr=args.alpha_lr,
            use_automatic_entropy_tuning=not args.no_automatic_entropy_tuning,),
    )
    torch.manual_seed(args.seed)

    # initialize with bc
    if args.bc_model is not None:
        variant['bc_model'] = 'models/{}'.format(args.bc_model)

    # timestapms
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    # bonus and the type
    if not args.offline:
        exp_dir = '{}/bonus_{}/{}_{}'.format(args.env,
                                             timestamp, args.bonus_type, args.seed)
        # use bonus in actor, critic or both
        if args.bonus_type == 'actor-critic':

            variant["use_bonus_policy"] = True
            variant["use_bonus_critic"] = True

        elif args.bonus_type == 'critic':

            variant["use_bonus_critic"] = True

        else:
            variant["use_bonus_policy"] = True

        exp_dir = '{0}_{1:.2g}'.format(exp_dir, args.beta)

    else:
        exp_dir = '{}/offline/{}_{}'.format(args.env, timestamp, args.seed)

    # setup the logger
    setup_logger(variant=variant, log_dir='logs/{}'.format(exp_dir))

    # cuda setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        # optionally set the GPU (default=False)
        ptu.set_gpu_mode(True, gpu_id=args.device_id)
        print('using gpu:{}'.format(args.device_id))

    else:
        map_location = 'cpu'
        ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)

    print('experiment dir:logs/{}'.format(exp_dir))
    experiment(variant)
