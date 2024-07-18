#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from collections import deque
import random
from model.actorNetwork import ActorNetwork
from model.criticNetwork import CriticNetwork
from tensorflow.keras.models import clone_model, load_model
from DDPG_agent import ReplayBuffer, DDPGAgent
from off_learning_env import OffLearningEnv
import os

# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Directory to save the models
checkpoint_dir = 'off_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load actor and critic networks
actor = ActorNetwork(input_shape=(224, 224, 3))
critic = CriticNetwork(input_shape=(224, 224, 3))

# Clone actor and critic networks to create target networks
target_actor = clone_model(actor.model)
target_actor.set_weights(actor.model.get_weights())

target_critic = clone_model(critic.model)
target_critic.set_weights(critic.model.get_weights())

# Initialize replay buffer and agent
buffer = ReplayBuffer(buffer_size=10000)
agent = DDPGAgent(actor.model, critic.model, target_actor, target_critic, buffer, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005)

# Function to save models
def save_models(actor, critic, episode):
    actor.model.save(os.path.join(checkpoint_dir, f'actor_{episode}.h5'))
    critic.model.save(os.path.join(checkpoint_dir, f'critic_{episode}.h5'))

# Function to load models
def load_models(actor, critic, episode):
    actor.model = load_model(os.path.join(checkpoint_dir, f'actor_{episode}.h5'))
    critic.model = load_model(os.path.join(checkpoint_dir, f'critic_{episode}.h5'))
    return actor, critic

# Load existing models if available
if os.path.exists(os.path.join(checkpoint_dir, 'actor_latest.h5')) and os.path.exists(os.path.join(checkpoint_dir, 'critic_latest.h5')):
    actor, critic = load_models(actor, critic, 'latest')
    print("Loaded existing models from checkpoints.")
    start_episode = int(np.max([int(f.split('_')[1].split('.')[0]) for f in os.listdir(checkpoint_dir) if 'actor_' in f])) + 1
else:
    start_episode = 0

# Initialize environment
data_dir = '/home/innodriver/InnoDriver_ws/src/dQN_modelTraining/offPolicyLearningData'
csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
envs = [OffLearningEnv(csv_file) for csv_file in csv_files]

# Training parameters
num_episodes = 1000
batch_size = 64
save_every = 50  # Save models every 50 episodes

for episode in range(start_episode, num_episodes):
    env = random.choice(envs)
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        
        buffer.add(state, action, reward, next_state, done)
        agent.train(batch_size)

        state = next_state
        episode_reward += reward

        if done:
            print(f"Episode: {episode}, Reward: {episode_reward}")
            break

    # Save models at specified intervals
    if episode % save_every == 0 or episode == num_episodes - 1:
        save_models(actor, critic, episode)
        save_models(actor, critic, 'latest')
        print(f"Models saved at episode {episode}")
