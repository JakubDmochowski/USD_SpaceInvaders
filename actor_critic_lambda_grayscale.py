import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.enable_eager_execution()

algorithm_name = "actor_critic_lambda"
version = "1"

load_snapshot = False # "./model_snapshots/model_actor_critic_lambda_1_ep95"
episode_count = 0
num_episodes = 100

seed = 9
gamma = 0.99 #discount factor
_lambda = 0.9
max_steps_per_episode = 10000
env = gym.make('SpaceInvaders-v0')
env.seed(seed)
eps = np.finfo(np.float32).eps.item()

x = 210
y = 160
num_inputs = x*y
hidden_layer_1_units = 64
num_actions = 6

inputs = layers.Input(shape=(num_inputs,))


model = None

if(load_snapshot):
    model = keras.models.load_model(load_snapshot)
else:
    hidden_layer_1 = layers.Dense(hidden_layer_1_units, activation="relu")(inputs)
    normalized = layers.LayerNormalization()(hidden_layer_1)
    actions = layers.Dense(num_actions, activation="softmax")(normalized)
    critic = layers.Dense(1)(normalized)
    model = keras.Model(inputs=inputs, outputs=[actions, critic])

actor_optimizer = keras.optimizers.Adam(learning_rate=0.01)
critic_optimizer = keras.optimizers.Adam(learning_rate=0.005)

critic_value_history = []
action_probs_history = []
rewards_history = []
reward_penalty = 0

def map_to_grayscale(state):
    Y = range(len(state))
    X = range(len(state[0]))
    C = range(len(state[0][0]))
    mapped = np.array([[0 for col in X] for row in Y])
    for y in Y:
        for x in X:
            val = 0
            for c in C:
                val += state[y][x][c]
            mapped[y][x] = val / 3
    return mapped

while episode_count in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        done = False
        while not done:
            env.render()
            
            state = map_to_grayscale(state)
            state = np.reshape(state, num_inputs)
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
        
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
        
            prev_state = state
            state, reward, done, _ = env.step(action)
            
            rewards_history.append(reward - reward_penalty)
            episode_reward += (reward - reward_penalty)
        
        time_diffs = []
        for i in range(len(rewards_history)):
            td = 0
            if i == len(rewards_history) - 1:
                td = rewards_history[i] - critic_value_history[i]
            else:
                td = rewards_history[i] + gamma * critic_value_history[i+1] - critic_value_history[i]
            time_diffs.append(td)
        
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        
        history = zip(action_probs_history, critic_value_history, returns, time_diffs)
        actor_losses = []
        critic_losses = []
        actor_loss = None
        critic_loss = None
        huber_loss = keras.losses.Huber()
        
        for log_prob, value, ret, td in history:
            if actor_loss is None:
                actor_loss = -log_prob * (ret-value)
                critic_loss = huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            else:
                actor_loss = _lambda * gamma * actor_loss - log_prob * (ret-value)
                critic_loss = _lambda * gamma * critic_loss + huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            print("actor_loss: {} | critic_loss: {}".format(actor_loss, critic_loss))
            actor_losses.append(actor_loss * td)
            critic_losses.append(critic_loss * td)
        actor_grads = actor_tape.gradient(actor_losses, model.trainable_variables)
        critic_grads = critic_tape.gradient(critic_losses, model.trainable_variables)
                        
        actor_optimizer.apply_gradients(zip(actor_grads, model.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_grads, model.trainable_variables))
        

        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        
    print("episode {}: rewarded with {}".format(episode_count, episode_reward))
    episode_count += 1
    if(not episode_count % 5):
        model.save('./model_snapshots/model_{}_{}_ep{}'.format(algorithm_name, version, episode_count))
