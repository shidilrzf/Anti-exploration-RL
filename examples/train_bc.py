from gym.envs.mujoco import HalfCheetahEnv
from rlkit.torch.networks import Mlp

import gym
import d4rl
import numpy as np

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader


from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


def train(network, dataloader, optimizer, epoch, device):

    loss_func = nn.MSELoss(reduction='sum')

    network.train()
    desc = 'Train'

    tqdm_bar = tqdm(dataloader)
    total_loss = 0
    for batch_idx, (obs, act) in enumerate(tqdm_bar):
        batch_size = obs.size(0)

        obs = obs.to(device)
        act = act.to(device)

        action_hat = network(obs)

        loss = loss_func(action_hat, act)

        network.zero_grad()
        loss.backward()
        optimizer.step()

        # Reporting
        total_loss += loss.item()

        tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format(desc, epoch, loss.item() / batch_size))

    return total_loss / (len(dataloader.dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='bc')
    parser.add_argument("--env", type=str, default='walker2d-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    # network
    parser.add_argument('--layer-size', default=256, type=int)
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
        device = torch.device("cuda")
        torch.cuda.set_device(args.device_id)
        print('using GPU')
    else:
        device = torch.device("cpu")

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

    # prepare networks
    M = args.layer_size
    network = Mlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
        output_activation=F.tanh,
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    epch = 0

    if args.load_model is not None:
        if os.path.isfile(args.load_model):
            checkpoint = torch.load(args.load_model)
            network.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            t_loss = checkpoint['train_loss']
            epch = checkpoint['epoch']
            print('Loading model: {}. Resuming from epoch: {}'.format(args.load_model, epch))
        else:
            print('Model: {} not found'.format(args.load_model))

    best_loss = np.Inf
    for epoch in range(epch, args.epochs):
        t_loss = train(network, dataloader, optimizer, epoch, device)
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
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': t_loss
            }, file_name)
