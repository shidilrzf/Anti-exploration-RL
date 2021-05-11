from gym.envs.mujoco import HalfCheetahEnv
from rlkit.torch.networks import Conditional_AE, Emdedding_AE, VAE

import gym
import d4rl
import numpy as np

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


def train(network, dataloader, optimizer, epoch, use_cuda):

    # loss_func = nn.MSELoss(reduction='sum')

    network.train()
    desc = 'Train'

    total_loss = 0

    tqdm_bar = tqdm(dataloader)
    for batch_idx, (obs, act) in enumerate(tqdm_bar):
        batch_loss = 0

        obs = obs.cuda() if use_cuda else obs
        act = act.cuda() if use_cuda else act

        # data = torch.cat((obs, act), dim=1)

        if args.network == 'VAE':
            predicted_act, mean, std = network(obs, act)
            recon_loss = F.mse_loss(predicted_act, act)
            KL_loss = - 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            loss = recon_loss + 0.5 * KL_loss
        else:
            predicted_act = network(obs, act)
            loss = F.mse_loss(predicted_act, act)

        network.zero_grad()
        loss.backward()
        optimizer.step()

        # Reporting
        batch_loss = loss.item() / obs.size(0)
        total_loss += loss.item() / obs.size(0)

        tqdm_bar.set_description('{} Epoch: [{}] Batch Loss: {:.2g}'.format(desc, epoch, batch_loss))

    return total_loss / (batch_idx + 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AE')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    # network
    parser.add_argument('--network', type=str, default='Conditional_AE', help='type of AE')
    parser.add_argument('--layer_size', default=750, type=int)
    parser.add_argument('--latent_size', default=12, type=int)
    parser.add_argument('--act_embed_size', default=16, type=int)
    parser.add_argument('--obs_embed_size', default=16, type=int)
    # Optimizer
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 2e-4')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input training batch-size')
    parser.add_argument('--seed', default=0, type=int)
    # normalization
    parser.add_argument('--use_norm', action='store_true', default=False, help='use norm')
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    parser.add_argument('--device-id', type=int, default=0, help='GPU device id (default: 0')
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    print('env:{}, action_dim:{}, obs_dim:{}'.format(args.env, action_dim, obs_dim))

    # timestamps
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    # preparing data and dataset
    # ds = env.get_dataset()
    ds = d4rl.qlearning_dataset(env)
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
    use_tb = False
    if use_tb:
        log_dir = 'runs/'
        logger = SummaryWriter(comment='_' + args.env + '_rnd')

    # parameters
    variant = dict(
        env=args.env,
        AE_type=args.network,
        lr=args.lr,
        use_norm=args.use_norm,
        batch_size=args.batch_size,
        seed=args.seed,)
    # prepare networks
    M = args.layer_size

    if args.network == 'Emdedding_AE':
        network = Emdedding_AE(
            input_sizes=[obs_dim, action_dim],
            embedding_sizes=[args.obs_embed_size, args.act_embed_size],
            hidden_sizes=[M, M],
        ).to(device)
        variant['obs_embed_size'] = args.obs_embed_size
        variant['act_embed_size'] = args.act_embed_size
        variant['hidden_size'] = M

    elif args.network == 'Conditional_AE':
        network = Conditional_AE(
            input_sizes=[obs_dim, action_dim],
            embedding_sizes=[args.obs_embed_size, args.act_embed_size],
            latent_size=args.latent_size,
            hidden_sizes=[M, M],
        ).to(device)

        variant['obs_embed_size'] = args.obs_embed_size
        variant['act_embed_size'] = args.act_embed_size
        variant['latent_size'] = args.latent_size
        variant['hidden_size'] = M

    elif args.network == 'VAE':
        network = VAE(
            input_sizes=[obs_dim, action_dim],
            latent_size=2 * action_dim,
        ).to(device)
        variant['latent_size'] = 2 * action_dim

    else:
        raise ValueError('Not implemented error')

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    print(variant)

    best_loss = np.Inf
    for epoch in range(args.epochs):
        t_loss = train(network, dataloader, optimizer, epoch, use_cuda)

        if use_tb:
            logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        if t_loss < best_loss:
            best_loss = t_loss
            file_name = 'models/AE_{}_{}.pt'.format(timestamp, args.env)
            print('Writing model checkpoint, loss:{:.2g}'.format(t_loss))
            print('Writing model checkpoint : {}'.format(file_name))

            torch.save({
                'epoch': epoch + 1,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': t_loss,
                'variant': variant
            }, file_name)
