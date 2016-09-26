import gym
from ACRL import ACRL

env = gym.make('CartPole-v0')
env.monitor.start('/tmp/mountaincar-exp-1', force=True)

#gamma, alphaR, alphaV, alphaU, lmbda
agent = ACRL(1, 1, 0.01, 0.001, 0.8, 1000, env)

for i_episode in range(500):
	observation = env.reset()
	for t in range(5000):
		env.render()
		
		# select an action
		action = agent.getAction(observation)
		
		# take a step in the environment
		observationNext, reward, done, info = env.step(action)
		
		# break if done
		if done:
			print("Episode " + str(i_episode) + " finished after {} timesteps".format(t+1))
			break
		
		# update the agent with the reward and next observation
		agent.update(observation, action, reward, observationNext)

		# update the observation
		observation = observationNext
env.monitor.close()