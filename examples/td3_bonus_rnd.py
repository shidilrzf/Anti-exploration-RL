

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
# from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector


from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

# from rlkit.torch.sac.sac_cls import SAC_BonusTrainer

from rlkit.torch.td3.td3_bonus_rnd_add import TD3_RND_ADD_Trainer
from rlkit.torch.td3.td3 import TD3Trainer

from rlkit.torch.networks import Mlp

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from torch.nn import functional as F


import argparse

import gym
import d4rl
import numpy as np
import torch
import time


def load_hdf5(dataset, replay_buffer, max_size):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], max_size)

    _obs = all_obs[:N - 1]
    _actions = all_act[:N - 1]
    _next_obs = all_obs[1:]
    _rew = np.squeeze(dataset['rewards'][:N - 1])
    _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
    _done = np.squeeze(dataset['terminals'][:N - 1])
    _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

    max_length = 1000
    ctr = 0
    # Only for MuJoCo environments
    # Handle the condition when terminal is not True and trajectory ends due to a timeout
    for idx in range(_obs.shape[0]):
        if ctr >= max_length - 1:
            ctr = 0
        else:
            replay_buffer.add_sample_only(
                _obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
            ctr += 1
            if _done[idx][0]:
                ctr = 0
    ###

    print (replay_buffer._size, replay_buffer._terminals.shape)


def experiment(variant):
    eval_env = gym.make(variant['env'])
    expl_env = eval_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    # q and policy netwroks
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )

    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )

    # if bonus: define bonus networks
    if not variant['offline']:
        M_bonus = variant['bonus_layer_size']
        feature_bonus = variant['bonus_feature_size']

        bonus_network = Mlp(
            input_size=obs_dim + action_dim,
            output_size=feature_bonus,
            hidden_sizes=[M_bonus, M_bonus, M_bonus, M_bonus]).to(ptu.device)

        bonus_target_network = Mlp(
            input_size=obs_dim + action_dim,
            output_size=feature_bonus,
            hidden_sizes=[M_bonus, M_bonus]).to(ptu.device)

        checkpoint = torch.load(variant['bonus_path'], map_location=ptu.device)
        bonus_network.load_state_dict(checkpoint['network_state_dict'])
        bonus_target_network.load_state_dict(checkpoint['target_state_dict'])
        print('Loading bonus model: {}'.format(variant['bonus_path']))

    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    dataset = eval_env.unwrapped.get_dataset()

    load_hdf5(dataset, replay_buffer, max_size=variant['replay_buffer_size'])

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
    if variant['offline']:
        trainer = TD3Trainer(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['trainer_kwargs']
        )
        print('Agent of type offline TD3 created')

    elif variant['bonus'] == 'bonus_add':
        trainer = TD3_RND_ADD_Trainer(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            bonus_network=bonus_network,
            bonus_target_network=bonus_target_network,
            beta=variant['bonus_beta'],
            use_bonus_critic=variant['use_bonus_critic'],
            use_bonus_policy=variant['use_bonus_policy'],
            use_log=variant['use_log'],
            bonus_norm_param=bonus_norm_param,
            rewards_shift_param=rewards_shift_param,
            device=ptu.device,
            **variant['trainer_kwargs']
        )
        print('Agent of type TD3 + additive bonus created')

    # elif variant['bonus'] == 'bonus_mlt':
    #     trainer = TD3_Bonus_MLT_Trainer(
    #         policy=policy,
    #         qf1=qf1,
    #         qf2=qf2,
    #         target_qf1=target_qf1,
    #         target_qf2=target_qf2,
    #         target_policy=target_policy,
    #         bonus_network=bonus_network,
    #         beta=variant['bonus_beta'],
    #         use_bonus_critic=variant['use_bonus_critic'],
    #         use_bonus_policy=variant['use_bonus_policy'],
    #         use_log=variant['use_log'],
    #         bonus_norm_param=bonus_norm_param,
    #         rewards_shift_param=rewards_shift_param,
    #         device=ptu.device,
    #         **variant['trainer_kwargs']
    #     )
    #     print('Agent of type TD3 + multiplicative bonus created')

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
        q_learning_alg=False,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TD3_bonus')
    parser.add_argument("--env", type=str, default='walker2d-medium-v0')
    # sac
    parser.add_argument('--alpha_lr', default=3e-5, type=float)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--num_samples', default=100, type=int)

    # bonus
    parser.add_argument('--offline', action='store_true', default=False, help='offline TD3')
    parser.add_argument('--bonus', type=str, default='bonus_add', help='different type of bonus: bonus_add, bonus_mlt')  # Q + bonus or Q * bonus
    parser.add_argument('--beta', default=1, type=float, help='beta for the bonus')
    parser.add_argument("--root_path", type=str, default='/home/shideh/', help='path to the bonus model')
    parser.add_argument("--bonus_model", type=str, default=None, help='name of the bonus model')
    parser.add_argument('--bonus_type', type=str, default='actor-critic', help='use bonus in actor, critic or both')
    parser.add_argument('--bonus_layer', default=128, type=int, help='layer size of the bonus model')
    parser.add_argument('--bonus_feature', default=4, type=int, help='feature size of the bonus model')
    parser.add_argument('--normalize', action='store_true', default=True, help='use normalization in bonus')
    parser.add_argument('--reward_shift', default=None, type=int, help='minimum reward')
    parser.add_argument('--use_log', action='store_true', default=False, help='use log(bonus(s, a) otherwise bonus(s, a)')

    # d4rl
    parser.add_argument('--dataset_path', type=str, default=None, help='d4rl dataset path')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device-id', type=int, default=0, help='GPU device id (default: 0')
    args = parser.parse_args()

    # noinspection
    # bonus_path = '{}RL/continuous_rnd/sac/examples/models/{}'.format(
    #     args.root_path, args.bonus_model)
    bonus_path = 'models/{}'.format(args.bonus_model)
    print(bonus_path)

    variant = dict(
        algorithm="TD3",
        version="normal",
        # bonus
        offline=args.offline,
        bonus=args.bonus,
        bonus_path=bonus_path,
        bonus_beta=2e4 * args.beta,
        use_log=args.use_log,
        layer_size=256,
        bonus_layer_size=args.bonus_layer,
        bonus_feature_size=args.bonus_feature,
        replay_buffer_size=int(1E6),
        buffer_filename=args.env,  # halfcheetah_101000.pkl',
        load_buffer=True,
        env=args.env,
        seed=args.seed,
        # bonus_type
        use_bonus_policy=False,
        use_bonus_critic=False,
        # use normalization for bonus
        normalize=args.normalize,

        # make reward positive
        reward_shift=args.reward_shift,

        algorithm_kwargs=dict(
            num_epochs=3000,
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
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    )

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
