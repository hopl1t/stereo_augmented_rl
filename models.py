import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class CommonActorCritic(nn.Module):
    """
    First FC layer is common between both nets
    """
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(CommonActorCritic, self).__init__()
        self.num_actions = num_actions
        obs_shape = obs_shape[0][0] * obs_shape[0][1]
        self.common_linear = nn.Linear(obs_shape, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state[0].flatten()).float().unsqueeze(0).to(self.device)
        common = F.leaky_relu(self.common_linear(state))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)

        return value, policy_dist


class ConvActorCritic(nn.Module):
    """
    Uses convolutional layers instead of fully connected layers
    """
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), kernels=(7,4,3),
                 channels=(32, 64, 64), strides=(4, 2, 1), pools=(1, 1, 1), **kwargs):
        super(ConvActorCritic, self).__init__()
        self.num_actions = num_actions
        self.pools = pools
        y_dim = obs_shape[0][0]
        x_dim = obs_shape[0][1]
        # output shape: (10, y - 3 + 1, x - 3 + 1), after pooling (10, (y - 3 + 1) // 2, (x - 3 + 1) // 2)
        self.conv1 = nn.Conv2d(1, channels[0], kernels[0], stride=strides[0])
        # output shape: (5, (y - 3 + 1) // 2 - 3 + 1, (x - 3 + 1) // 2 - 3 + 1), no pooling
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1], stride=strides[1])
        # output shape: (5, (y - 3 + 1) // 2 - 6 + 2, (x - 3 + 1) // 2 - 6 + 2),
        # after pooling (5, ((y - 3 + 1) // 2 - 6 + 2) // 2, ((x - 3 + 1) // 2 - 6 + 2) // 2)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2], stride=strides[2])
        vid_feature_size = ((((((((y_dim - kernels[0]) // strides[0] + 1) // pools[0]) - kernels[1]) // strides[1] + 1) // pools[1] - kernels[2]) // strides[2] + 1) // pools[2]) * \
                           ((((((((x_dim - kernels[0]) // strides[0] + 1) // pools[0]) - kernels[1]) // strides[1] + 1) // pools[1] - kernels[2]) // strides[2] + 1) // pools[2]) * channels[2]
        self.common_linear = nn.Linear(vid_feature_size, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions)
        self.device = device
        # utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state[0]).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vid_feature = F.max_pool2d(F.relu(self.conv1(state)), self.pools[0])
        vid_feature = F.relu(self.conv2(vid_feature))
        vid_feature = F.max_pool2d(F.relu(self.conv3(vid_feature)), self.pools[2])
        # vid_feature = torch.flatten(vid_feature, start_dim=1)
        vid_feature = vid_feature.view(vid_feature.shape[0], -1)
        common = F.relu(self.common_linear(vid_feature))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)
        return value, policy_dist


class NaiveSoundActorCritic(nn.Module):
    """
    Naive multimodal net that processes sound and video separately before concating into common layer
    Naive in the sense that no RNN is applied over time
    This net can take any naive multimodal input such as buffer, max volume or frequency + volume (Fourire).
    """
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(NaiveSoundActorCritic, self).__init__()
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.common_video_linear = nn.Linear(obs_shape[0], hidden_size)
        self.common_audio_linear = nn.Linear(obs_shape[1], hidden_size // 4)
        self.critic_linear = nn.Linear(hidden_size + hidden_size // 4, 1)
        self.actor_linear = nn.Linear(hidden_size + hidden_size // 4, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        video_state = torch.from_numpy(state[0]).float().unsqueeze(0).to(self.device)
        audio_state = torch.from_numpy(state[1]).float().unsqueeze(0).to(self.device)
        common_video = F.leaky_relu(self.common_video_linear(video_state))
        common_audio = F.leaky_relu(self.common_audio_linear(audio_state))
        common = torch.cat((common_video, common_audio))
        value = self.critic_linear(common)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)
        return value, policy_dist


class SimpleDQN(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(SimpleDQN, self).__init__()
        obs_shape = obs_shape[0][0] * obs_shape[0][1]
        self.net = nn.Sequential(nn.Linear(obs_shape, hidden_size // 2), nn.LeakyReLU(),
                                 nn.Linear(hidden_size // 2, hidden_size), nn.LeakyReLU(),
                                 nn.Linear(hidden_size, num_actions))
        self.device = device
        self.num_actions = num_actions
        utils.init_weights(self)

    def forward(self, state):
        # this happens when the state comes from the experience dataloader
        if isinstance(state[0], torch.Tensor):
            state = state[0].flatten(start_dim=1).float().to(self.device)
        else:
            state = torch.from_numpy(state[0].flatten()).float().unsqueeze(0).to(self.device)
        return self.net(state)


class ConvDQN(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), **kwargs):
        super(ConvDQN, self).__init__()
        y_dim = obs_shape[0][0]
        x_dim = obs_shape[0][1]
        # output shape: (10, y - 3 + 1, x - 3 + 1), after pooling (10, (y - 3 + 1) // 2, (x - 3 + 1) // 2)
        self.conv1 = nn.Conv2d(1, 10, 3)
        # output shape: (5, (y - 3 + 1) // 2 - 3 + 1, (x - 3 + 1) // 2 - 3 + 1), no pooling
        self.conv2 = nn.Conv2d(10, 5, 3)
        # output shape: (5, (y - 3 + 1) // 2 - 6 + 2, (x - 3 + 1) // 2 - 6 + 2),
        # after pooling (5, ((y - 3 + 1) // 2 - 6 + 2) // 2, ((x - 3 + 1) // 2 - 6 + 2) // 2)
        self.conv3 = nn.Conv2d(5, 16, 3)
        vid_feature_size = (((((y_dim - 3 + 1) // 2) - 3 + 1) - 3 + 1) // 2) * \
                           (((((x_dim - 3 + 1) // 2) - 3 + 1) - 3 + 1) // 2) * 16

        self.fc1 = nn.Linear(vid_feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.device = device
        self.num_actions = num_actions
        utils.init_weights(self)

    def forward(self, state):
        if isinstance(state[0], torch.Tensor):
            state = state[0].float().unsqueeze(1).to(self.device)
        else:
            state = torch.from_numpy(state[0]).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vid_feature = F.max_pool2d(F.leaky_relu(self.conv1(state)), 2)
        vid_feature = F.leaky_relu(self.conv2(vid_feature))
        vid_feature = F.max_pool2d(F.leaky_relu(self.conv3(vid_feature)), 2)
        vid_feature = torch.flatten(vid_feature, start_dim=1)
        fc1 = F.leaky_relu(self.fc1(vid_feature))
        return self.fc2(fc1)


class ConvLSTMActorCritic(nn.Module):
    """
    Uses convolutional layers instead of fully connected layers
    """
    def __init__(self, obs_shape, num_actions, hidden_size=512, device=torch.device('cpu'), n_layers=4, **kwargs):
        super(ConvLSTMActorCritic, self).__init__()
        self.num_actions = num_actions
        y_dim = obs_shape[0][0]
        x_dim = obs_shape[0][1]
        # output shape: (10, y - 3 + 1, x - 3 + 1), after pooling (10, (y - 3 + 1) // 2, (x - 3 + 1) // 2)
        self.conv1 = nn.Conv2d(1, 10, 3)
        # output shape: (5, (y - 3 + 1) // 2 - 3 + 1, (x - 3 + 1) // 2 - 3 + 1), no pooling
        self.conv2 = nn.Conv2d(10, 5, 3)
        # output shape: (5, (y - 3 + 1) // 2 - 6 + 2, (x - 3 + 1) // 2 - 6 + 2),
        # after pooling (5, ((y - 3 + 1) // 2 - 6 + 2) // 2, ((x - 3 + 1) // 2 - 6 + 2) // 2)
        self.conv3 = nn.Conv2d(5, 16, 3)
        vid_feature_size = (((((y_dim - 3 + 1) // 2) - 3 + 1) - 3 + 1) // 2) * \
                           (((((x_dim - 3 + 1) // 2) - 3 + 1) - 3 + 1) // 2) * 16
        self.common_linear = nn.Linear(vid_feature_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size // 4, num_layers=self.n_layers, bidirectional=False)
        self.h0 = torch.zeros(n_layers, 1, hidden_size, device=device)
        self.critic_linear = nn.Linear(hidden_size // 4, 1)
        self.actor_linear = nn.Linear(hidden_size, num_actions)
        self.device = device
        utils.init_weights(self)

    def forward(self, state):
        state = torch.from_numpy(state[0]).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vid_feature = F.max_pool2d(F.leaky_relu(self.conv1(state)), 2)
        vid_feature = F.leaky_relu(self.conv2(vid_feature))
        vid_feature = F.max_pool2d(F.leaky_relu(self.conv3(vid_feature)), 2)
        vid_feature = torch.flatten(vid_feature, start_dim=1)
        common = F.leaky_relu(self.common_linear(vid_feature))
        lstm_out, _ = self.lstm(common, self.h0)
        value = self.critic_linear(lstm_out)
        policy_dist = F.softmax(self.actor_linear(common), dim=1)
        return value, policy_dist
