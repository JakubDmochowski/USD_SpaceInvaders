import numpy as np 
import gym
import json

def to_state_index(state):
	result = 0
	for s in state:
		result = result<<8
		result = result | s
	return hash(result) % 10_000_000



#Function to choose the next action 
def choose_action(state): 
	action=0
	index = to_state_index(state)
	if np.random.uniform(0, 1) < epsilon: 
		action = env.action_space.sample() 
	else: 
		action = np.argmax(Q[index, :]) 
	return action 

#Function to learn the Q-value 
def update(state, state2, reward, action, action2): 
	i1 = to_state_index(state)
	i2 = to_state_index(state2)
	predict = Q[i1, action] 
	target = reward + gamma * Q[i2, action2] 
	Q[i1, action] = Q[i1, action] + alpha * (target - predict) 

env = gym.make('SpaceInvaders-ram-v0') 

#Defining the different parameters 
epsilon = 0.9
total_episodes = 10000
max_steps = 100000
alpha = 0.85
gamma = 0.95


#Initializing the Q-matrix 
Q = np.zeros([10_000_000, env.action_space.n]) 


#Initializing the reward 
reward=0

# Starting the SARSA learning 
for episode in range(total_episodes): 
	t = 0
	state1 = env.reset()
	action1 = choose_action(state1)
	while True: 
		#Visualizing the training 
		env.render() 
		
		#Getting the next state 
		state2, reward, done, info = env.step(action1) 

		#Choosing the next action 
		action2 = choose_action(state2) 
		
		#Learning the Q-value 
		print(to_state_index(state2))
		update(state1, state2, reward, action1, action2) 

		state1 = state2 
		action1 = action2 
		
		#Updating the respective vaLues 
		t += 1
		reward += 1
		
		#If at the end of learning process 
		if done: 
			break
	print(reward)
	if episode %100 == 0:
		f = open(str(episode) + '.json', 'w')
		f.write(json.dumps(Q.tolist()))
		f.close()
#Evaluating the performance 
print ("Performace : ", reward/total_episodes) 

#Visualizing the Q-matrix 
print(Q) 
