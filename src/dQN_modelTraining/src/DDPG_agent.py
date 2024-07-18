#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, actor_model, critic_model, target_actor_model, target_critic_model, buffer, actor_lr, critic_lr, gamma, tau):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor_model = target_actor_model
        self.target_critic_model = target_critic_model
        self.buffer = buffer
        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = Adam(actor_lr)
        self.critic_optimizer = Adam(critic_lr)

    def update_target_network(self, target_weights, weights):
        for (target, source) in zip(target_weights, weights):
            target.assign(self.tau * source + (1 - self.tau) * target)

    def policy(self, state):
        state = np.expand_dims(state, axis=0)
        return self.actor_model(state)[0]

    def train(self, batch_size):
        if self.buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_model(next_states)
            y = rewards + self.gamma * (1 - dones) * self.target_critic_model([next_states, target_actions])
            critic_value = self.critic_model([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
        
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions = self.actor_model(states)
            critic_value = self.critic_model([states, actions])
            actor_loss = -tf.reduce_mean(critic_value)
        
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target_network(self.target_actor_model.variables, self.actor_model.variables)
        self.update_target_network(self.target_critic_model.variables, self.critic_model.variables)
