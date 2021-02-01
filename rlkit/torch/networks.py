"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


# def identity(x):
#     return x


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=nn.Identity(),
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class Mlp_embedding(nn.Module):
    def __init__(
            self,
            input_sizes,
            embedding_sizes,
            hidden_sizes,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=nn.Identity(),
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_sizes = input_sizes
        self.output_size = output_size

        self.embedding_sizes = embedding_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
#         in_size = input_size

        self.embed1 = nn.Linear(input_sizes[0], embedding_sizes[0])
        self.embed2 = nn.Linear(input_sizes[1], embedding_sizes[1])

        in_size = embedding_sizes[0] + embedding_sizes[1]

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, act, return_preactivations=False):
        h1 = F.relu(self.embed1(obs))
        h2 = F.relu(self.embed2(act))

        h = torch.cat((h1, h2), dim=-1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


# class AE(nn.Module):
#     """
#     Simple AE model
#     """

#     def __init__(self, input_sizes, embedding_sizes, hidden_sizes, latent_size):
#         super(AE, self).__init__()

#         self.obs_sz, self.act_sz = input_sizes[0], input_sizes[1]
#         self.encoder = Mlp_embedding(input_sizes, embedding_sizes, hidden_sizes, latent_size)
#         self.decoder = Mlp(latent_size, hidden_sizes, self.act_sz)

#     def forward(self, obs, act):
#         z = self.encoder(obs, act)
#         act_hat = self.decoder(z)
#         return act_hat

class Conditional_AE(nn.Module):
    """
    Simple AE model
    """

    def __init__(self, input_sizes, embedding_sizes, hidden_sizes, latent_size):
        super().__init__()

        self.obs_sz, self.act_sz = input_sizes[0], input_sizes[1]
        self.encoder = Mlp_embedding(input_sizes, embedding_sizes, hidden_sizes, latent_size)
        self.decoder = Mlp(latent_size + self.obs_sz, hidden_sizes, self.act_sz)

    def forward(self, obs, act):
        z = self.encoder(obs, act)
        z = torch.cat((z, obs), 1)
        act_hat = self.decoder(z)

        return torch.tanh(act_hat)


class Emdedding_AE(nn.Module):
    """
    Simple AE model
    """

    def __init__(self, input_sizes, embedding_sizes, hidden_sizes):
        super().__init__()

        self.obs_sz, self.act_sz = input_sizes[0], input_sizes[1]
        self.obs_embedding = Mlp(self.obs_sz, hidden_sizes, embedding_sizes[0])
        self.act_embedding = Mlp(self.act_sz, hidden_sizes, embedding_sizes[1])
        self.decoder = Mlp(embedding_sizes[0] + embedding_sizes[1], hidden_sizes, self.act_sz)

    def forward(self, obs, act):
        z_obs = self.obs_embedding(obs)
        z_act = self.act_embedding(act)

        z = torch.cat((z_obs, z_act), 1)
        act_hat = self.decoder(z)
        return torch.tanh(act_hat)


class VAE(nn.Module):
    def __init__(self, input_sizes, latent_dim, M=750):
        super(VAE, self).__init__()

        obs_dim, action_dim = input_sizes[0], input_sizes[1]
        self.latent_dim = latent_dim

        self.enc_fc1 = nn.Linear(obs_dim + action_dim, M)
        self.enc_fc2 = nn.Linear(M, M)

        self.mean_fc = nn.Linear(M, latent_dim)
        self.log_var_fc = nn.Linear(M, latent_dim)

        self.dec_fc1 = nn.Linear(obs_dim + latent_dim, M)
        self.dec_fc2 = nn.Linear(M, M)
        self.dec_fc3 = nn.Linear(M, action_dim)

    def encode(self, obs, act):

        h = torch.cat([obs, action], 1)
        z = F.relu(self.enc_fc1())
        z = F.relu(self.enc_fc2(z))

        return z

    def decode(self, obs, z):
        h = torch.cat([obs, z], 1)
        h = F.relu(self.dec_fc1(h))
        h = F.relu(self.dec_fc2(a))

        return torch.tanh(self.dec_fc3(h))

    def forward(self, obs, act):

        z = self.encode(obs, act)

        mean = self.mean_fc(z)
        log_var = self.log_var_fc(z).clamp(-4, 15)
        std = torch.exp(log_var)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std
