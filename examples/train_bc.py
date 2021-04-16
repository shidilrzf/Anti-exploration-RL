from gym.envs.mujoco import HalfCheetahEnv

import gym
import d4rl
import numpy as np

import argparse
import torch
import torch.optim as optim
# import torch.nn as nn
# from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader
from rlkit.torch.sac.policies import TanhGaussianPolicy
import rlkit.torch.pytorch_util as ptu


from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


def train(policy, dataloader, optimizer, epoch):

    policy.train()
    desc = 'Train'

    tqdm_bar = tqdm(dataloader)
    total_loss = 0
    for batch_idx, (obs, actions) in enumerate(tqdm_bar):
        batch_size = obs.size(0)

        obs = obs.to(ptu.device)
        actions = actions.to(ptu.device)

        _, _, _, log_pi, *_ = policy(obs, reparameterize=True, return_log_prob=True)
        data_log_pi = policy.log_prob(obs, actions)
        policy_loss = (log_pi - data_log_pi).mean()

        policy.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Reporting
        total_loss += policy_loss.item()

        tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format(desc, epoch, policy_loss.item() / batch_size))

    return total_loss / (len(dataloader.dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='bc')
    parser.add_argument("--env", type=str, default='walker2d-medium-v0')
    # policy
    parser.add_argument('--layer-size', default=256, type=int)
    parser.add_argument('--mixture', action='store_true', default=False, help='use norm')

    # Optimizer
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 2e-4')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input training batch-size')
    parser.add_argument('--seed', default=0, type=int)
    # normalization
    parser.add_argument('--use_norm', action='store_true', default=False, help='use norm')
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    parser.add_argument('--device-id', type=int, default=0, help='GPU device id (default: 0')

    # tb
    parser.add_argument('--log-dir', type=str, default='runs', help='logging directory (default: runs)')

    # load model
    parser.add_argument('--load-model', type=str, default=None, help='Load model to resume training for (default None)')
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    # timestamps
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    print('-------------------------------')
    print('Env:{}, timestamp:{}'.format(args.env, timestamp))
    print('-------------------------------')

    # preparing data and dataset
    ds = env.get_dataset()
    obs = ds['observations']
    actions = ds['actions']
    if args.use_norm:
        print('.. using normalization ..')
        obs = (obs - obs.mean(axis=0)) / obs.std(axis=0)
        # actions = (actions - actions.mean(axis=0)) / actions.std(axis=0)

    dataset = TensorDataset(torch.Tensor(obs), torch.Tensor(actions))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if use_cuda:
        ptu.set_gpu_mode(True, gpu_id=args.device_id)
        print('using gpu:{}'.format(args.device_id))

    else:
        map_location = 'cpu'
        ptu.set_gpu_mode(False)

    # Setup asset directories
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('runs'):
        os.makedirs('runs')

    # Logger
    use_tb = args.log_dir is not None
    log_dir = args.log_dir
    if use_tb:
        logger = SummaryWriter(comment='_' + args.env + '_bc')

    # prepare policys
    M = args.layer_size
    if args.mixture:
        raise ValueError('Not implemented error')
    else:
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        ).to(ptu.device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    epch = 0

    if args.load_model is not None:
        if os.path.isfile(args.load_model):
            checkpoint = torch.load(args.load_model)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            t_loss = checkpoint['train_loss']
            epch = checkpoint['epoch']
            print('Loading model: {}. Resuming from epoch: {}'.format(args.load_model, epch))
        else:
            print('Model: {} not found'.format(args.load_model))

    best_loss = np.Inf
    for epoch in range(epch, args.epochs):
        t_loss = train(policy, dataloader, optimizer, epoch)
        print('=> epoch: {} Average Train loss: {:.4f}'.format(epoch, t_loss))

        if use_tb:
            logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        if t_loss < best_loss:
            best_loss = t_loss
            file_name = 'models/bc_{}_{}.pt'.format(timestamp, args.env)
            print('Writing model checkpoint, loss:{:.2g}'.format(t_loss))
            print('Writing model checkpoint : {}'.format(file_name))

            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': t_loss
            }, file_name)
