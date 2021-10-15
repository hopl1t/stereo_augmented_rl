import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class SimpleActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(SimpleActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        return value, policy_dist


class CommonActorCritic(nn.Module):
    """
    First FC layer is common between both nets
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(CommonActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common_linear = nn.Linear(num_inputs, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = F.leaky_relu(self.common_linear(state))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)

        return value, policy_dist


class DoubleCommonActorCritic(nn.Module):
    """
    First FC layer is common between both nets
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(DoubleCommonActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common_linear = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.LeakyReLU(),
                                           nn.Linear(hidden_size, hidden_size // 2), nn.LeakyReLU())
        self.critic_linear = nn.Linear(hidden_size // 2, 1)
        self.actor_linear = nn.Linear(hidden_size // 2, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common_linear(state)
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)
        return value, policy_dist


class ConvActorCritic(nn.Module):
    """
    Uses two common 3x3 convs on the first layer followed by a common linear.
    Expects a num_input of that is the dimension of one side of the input (ie 7).
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(ConvActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        # 1 input channel, 10 out channels, 3x3 square convolution
        # out shape will be: 7 - 3 + 2 + 1 = 10x7x7 (for a 7x7 input)
        # no pooling
        conv1 = nn.Conv2d(1, 10, 3, padding=1)
        # 10 input channels, 5 out channels, 3x3 square convolution
        # out shape will be: 7 - 3 + 2 + 1 = 5x7x7
        # no pooling
        conv2 = nn.Conv2d(10, 5, 3, padding=1)
        self.common_linear = nn.Linear((num_inputs**2) * 5, hidden_size)
        self.conv_block = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        convolved = self.conv_block(state.unsqueeze(0))
        common = F.relu(self.common_linear(torch.flatten(convolved, start_dim=1)))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)

        return value, policy_dist


class DiscreteActorCritic(nn.Module):
    """
    For a discretisized continouous environment with more than one action per state
    The difference from a regular net is that we do 2 softmaxs
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(DiscreteActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common_linear = nn.Linear(num_inputs, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions * num_discrete)
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = F.relu(self.common_linear(state))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common).view(self.num_actions, self.num_discrete), dim=1)

        return value, policy_dist


class TripleDiscreteActorCritic(nn.Module):
    """
    Like the prior only with 2 common layers
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(TripleDiscreteActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common_linear = nn.Sequential(nn.Linear(num_inputs, hidden_size//2), nn.ReLU(),
                                           nn.Linear(hidden_size//2, hidden_size), nn.ReLU(),
                                           nn.Linear(hidden_size, hidden_size//2), nn.ReLU())
        self.critic_linear = nn.Linear(hidden_size//2, 1)
        self.actor_linear = nn.Linear(hidden_size//2, num_actions * num_discrete)
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common_linear(state)
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common).view(self.num_actions, self.num_discrete), dim=1)
        return value, policy_dist


class SplitDiscreteActorCritic(nn.Module):
    """
    Both nets diverge and converge
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(SplitDiscreteActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.LeakyReLU()) #, nn.BatchNorm1d(hidden_size)
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(),
                                   nn.Linear(hidden_size//2, num_actions * num_discrete)) # , nn.BatchNorm1d(hidden_size//2)
        self.critic1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU()) # , nn.BatchNorm1d(hidden_size//2)
        self.critic2 = nn.Linear(hidden_size//2, 1)
        self.residual = nn.Sequential(nn.Linear(num_actions * num_discrete, hidden_size//2), nn.LeakyReLU())
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common(state)
        policy_dist = F.softmax(self.actor(common).view(self.num_actions, self.num_discrete), dim=1)
        critic1 = self.critic1(common)
        residual = self.residual(policy_dist.flatten())
        value = self.critic2(F.relu(critic1 + residual))
        return value, policy_dist


class GaussianActorCritic(nn.Module):
    """
    For a gaussian continouous environment with more than one action per state
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(GaussianActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common_linear = nn.Linear(num_inputs, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions * 2)
        self.std_bias = kwargs['std_bias']
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = F.relu(self.common_linear(state))
        value = self.critic_linear(common)
        policy = self.actor_linear(common)
        mu, sigma = torch.split(policy, policy.shape[-1]//2, 1)
        # softplus transformation (soft relu) and a -5 bias is added
        sigma = F.softplus(sigma - self.std_bias, beta=1)
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            raise

        return value, torch.stack((mu, sigma))


class DiscreteConvActorCritic(nn.Module):
    """
    For a discretisized continouous environment with more than one action per state
    The difference from a regular net is that we do 2 softmaxs
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(DiscreteConvActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs

        # 1 input channel, 30 out channels, 1x8 convolution
        # out shape will be: 8 - 8 + 2*4 + 1 = 9x20 (for a 1x8 input)
        conv1 = nn.Conv1d(1, 20, 8, padding=4)
        self.conv_block = nn.Sequential(conv1, nn.ReLU())
        common_linear1 = nn.Linear(9*20, hidden_size)
        self.common_linear = nn.Sequential(common_linear1, nn.ReLU())
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions * num_discrete)
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        convolved = self.conv_block(state.unsqueeze(0))
        common = self.common_linear(torch.flatten(convolved, start_dim=1))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common).view(self.num_actions, self.num_discrete), dim=1)

        return value, policy_dist


class GaussianConvActorCritic(nn.Module):
    """
    For a discretisized continouous environment with more than one action per state
    The difference from a regular net is that we do 2 softmaxs
    """
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(GaussianConvActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        # 1 input channel, 30 out channels, 1x8 convolution
        # out shape will be: 8 - 8 + 2*4 + 1 = 9x20 (for a 1x8 input)
        conv1 = nn.Conv1d(1, 20, 8, padding=4)
        self.conv_block = nn.Sequential(conv1, nn.ReLU())
        common_linear1 = nn.Linear(9*20, hidden_size)
        self.common_linear = nn.Sequential(common_linear1, nn.ReLU())
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions * num_discrete)
        self.num_discrete = num_discrete
        self.std_bias = kwargs['std_bias']
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        convolved = self.conv_block(state.unsqueeze(0))
        common = self.common_linear(torch.flatten(convolved, start_dim=1))
        value = self.critic_linear(common)
        policy = self.actor_linear(common)
        mu, sigma = torch.split(policy, policy.shape[-1] // 2, 1)
        # softplus transformation (soft relu) and a -5 bias is added
        sigma = F.softplus(sigma - self.std_bias, beta=1)
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            raise
        return value, torch.stack((mu, sigma))


class SplitDiscreteConvActorCritic(nn.Module):
    """
    Both nets diverge and converge
    The common layer is a conv layer followed by a linear one instead of just a linear
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(SplitDiscreteConvActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        out_channels = hidden_size // 25
        self.common = nn.Sequential(nn.Conv1d(1, out_channels, 8, padding=4), nn.LeakyReLU(), nn.Flatten(),
                                    nn.Linear(9 * out_channels, hidden_size), nn.LeakyReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(),
                                   nn.Linear(hidden_size//2, num_actions * num_discrete))
        self.critic1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU())
        self.critic2 = nn.Linear(hidden_size//2, 1)
        self.residual = nn.Sequential(nn.Linear(num_actions * num_discrete, hidden_size//2), nn.LeakyReLU())
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common(state.unsqueeze(0))
        policy_dist = F.softmax(self.actor(common).view(self.num_actions, self.num_discrete), dim=1)
        critic1 = self.critic1(common)
        residual = self.residual(policy_dist.flatten())
        value = self.critic2(F.relu(critic1 + residual))
        return value, policy_dist


class SplitGaussianActorCritic(nn.Module):
    """
    Both nets diverge and converge
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400, device=torch.device('cpu'), num_discrete=100, **kwargs):
        super(SplitGaussianActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.LeakyReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(),
                                   nn.Linear(hidden_size//2, num_actions * 2))
        self.critic1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU())
        self.critic2 = nn.Linear(hidden_size//2, 1)
        self.residual = nn.Sequential(nn.Linear(num_actions, hidden_size//2), nn.LeakyReLU())
        self.num_discrete = num_discrete
        self.std_bias = kwargs['std_bias']
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common(state)
        policy = self.actor(common)
        mu, sigma = torch.split(policy, policy.shape[-1] // 2, 1)
        critic1 = self.critic1(common)
        residual = self.residual(mu.flatten())
        value = self.critic2(F.relu(critic1 + residual))
        # softplus transformation (soft relu) and a -5 bias is added
        sigma = F.softplus(sigma - self.std_bias, beta=1)
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            raise
        return value, torch.stack((mu, sigma))


class SplitActorCritic(nn.Module):
    """
    Both nets diverge and converge
    This is the vanilla version with no discretization (for Sokoban for example)
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400, device=torch.device('cpu'), **kwargs):
        super(SplitActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.LeakyReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(),
                                   nn.Linear(hidden_size//2, num_actions))
        self.critic1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU())
        self.critic2 = nn.Linear(hidden_size//2, 1)
        self.residual = nn.Sequential(nn.Linear(num_actions, hidden_size//2), nn.LeakyReLU())
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common(state)
        policy_dist = F.softmax(self.actor(common), dim=1)
        critic1 = self.critic1(common)
        residual = self.residual(policy_dist.flatten())
        value = self.critic2(F.relu(critic1 + residual))
        return value, policy_dist


class DoubleSplitActorCritic(nn.Module):
    """
    Both nets diverge and converge
    This is the vanilla version with no discretization (for Sokoban for example)
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400, device=torch.device('cpu'), **kwargs):
        super(DoubleSplitActorCritic, self).__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.common = nn.Sequential(nn.Linear(num_inputs, hidden_size//2), nn.LeakyReLU(),
                                    nn.Linear(hidden_size//2, hidden_size), nn.LeakyReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(),
                                   nn.Linear(hidden_size//2, num_actions))
        self.critic1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU())
        self.critic2 = nn.Linear(hidden_size//2, 1)
        self.residual = nn.Sequential(nn.Linear(num_actions, hidden_size//2), nn.LeakyReLU())
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        common = self.common(state)
        policy_dist = F.softmax(self.actor(common), dim=1)
        critic1 = self.critic1(common)
        residual = self.residual(policy_dist.flatten())
        value = self.critic2(F.relu(critic1 + residual))
        return value, policy_dist


class SimpleDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(SimpleDQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_inputs, hidden_size // 2), nn.LeakyReLU(),
                                 nn.Linear(hidden_size // 2, hidden_size), nn.LeakyReLU(),
                                 nn.Linear(hidden_size, num_actions))
        self.device = device
        self.num_actions = num_actions
        utils.init_weights(self)

    def forward(self, state):
        if isinstance(state, torch.Tensor):
            state = state.float().to(self.device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.net(state)


class DiscretizedDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=512, device=torch.device('cpu'), num_discrete=10, **kwargs):
        super(DiscretizedDQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_inputs, hidden_size // 2), nn.LeakyReLU(),
                                 nn.Linear(hidden_size // 2, hidden_size), nn.LeakyReLU(),
                                 nn.Linear(hidden_size, num_actions * num_discrete))
        self.num_actions = num_actions
        self.num_discrete = num_discrete
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        if isinstance(state, torch.Tensor):
            state = state.float().to(self.device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.net(state).view(-1, self.num_actions, self.num_discrete).squeeze(0)
