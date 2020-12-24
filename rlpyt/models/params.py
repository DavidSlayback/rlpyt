# Convolutional format: in_channel, out_channel, num_filters, stride
CONVNET_MINIGRID_TINY = {
    'channels': [16, 32, 64],
    'kernel_sizes': [3, 2, 2],
    'strides': [1, 1, 1],
    'paddings': [0, 0, 0],
    'use_maxpool': False
}  # Gym-minigrid (sample-factory)

CONVNET_MINIGRID_ENSEMBLES = {
    'channels': [16, 16, 32],
    'kernel_sizes': [3, 2, 2],
    'strides': [2, 2, 2],
    'paddings': [1, 1, 0],
    'use_maxpool': False
}  # Gym-minigrid (https://arxiv.org/pdf/1906.10667.pdf)

CONVNET_A3C = {
    'channels': [16, 32],
    'kernel_sizes': [8, 4],
    'strides': [4, 2],
    'paddings': [0, 1],
    'use_maxpool': False}  # Impala/A3C/A2C for Atari

CONVNET_DOOM = {
    'channels': [32, 64, 128],
    'kernel_sizes': [8, 4, 3],
    'strides': [4, 2, 2],
    'paddings': [0, 0, 0],
    'use_maxpool': False}  # Impala/A3C/A2C for VizDoom

CONVNET_DQN = {
    'channels': [32, 64, 64],
    'kernel_sizes': [8, 4, 3],
    'strides': [4, 2, 1],
    'paddings': [0, 1, 1],
    'use_maxpool': False}  # DQN for Atari

MLP_DDPG = {
    'hidden_sizes': [400, 300]
}  # MLP for DDPG/TD3 Q and V

MLP_SAC = {
    'hidden_sizes': [256, 256]
}  # MLP for SAC

MLP_PPO = {
    'hidden_sizes': [64, 64]
}  # MLP for PPO

MLP_PPO_V = {
    'hidden_sizes': [128, 128]
}  # MLP for PPO, using a wider value network

ENCODER_MINIGRID = {
    'hidden_sizes': [128,128],
}