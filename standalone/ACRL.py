from pylab import *
import numpy as np
import random
import math
from kanerva import KanervaCoder

class ACRL():
	def __init__(self,gamma, alphaR, alphaV, alphaU, lmbda, n, env):
		self.gamma = gamma
		self.alphaR = alphaR
		self.alphaV = alphaV
		self.alphaU = alphaU
		self.lmbda = lmbda
		self.action = 0

		# define the state space
		self.lowerStateBound = np.array(env.observation_space.low)
		self.upperStateBound = np.array(env.observation_space.high)
		self.rangeState = self.upperStateBound - self.lowerStateBound
		self.stateDimensions = len(self.lowerStateBound)

		# define the action space
		if str(env.action_space)[:3] == 'Box':
			# continuous action space
			self.lowerActionBound = np.array(env.action_space.low)
			self.upperActionBound = np.array(env.action_space.high)
			self.isDiscrete = False
			self.rangeAction = np.subtract(self.upperActionBound, self.lowerActionBound)
		
		elif str(env.action_space)[:8] == 'Discrete':
			# discrete action space
			self.isDiscrete = True
			self.lowerActionBound = [0]
			self.upperActionBound = [env.action_space.n]

		self.rangeAction = np.subtract(self.upperActionBound, self.lowerActionBound)

		self.kcoding = KanervaCoder(n, self.stateDimensions)

		self.n = n
		
		self.avgR = 0
		self.ev = np.zeros(n)
		self.e_mu = np.zeros(n)
		self.e_sigma = np.zeros(n)
		
		self.w = np.zeros(n)
		self.u_mu = np.zeros(n)
		self.u_sigma = np.zeros(n)
		
		self.delta = 0.0
		self.R = 0.0
		self.value = 0.0
		self.nextValue = 0.0
		
		self.compatibleFeatures_mu = np.zeros(n)
		self.compatibleFeatures_sigma = np.zeros(n)
		
		self.mean = 0.0
		self.sigma = 1.0
    
	def Value(self,features):
		Val = 0.0
		for index in features:
			Val += self.w[index]
		self.value = Val
	
	def Next_Value(self,features):
		Val = 0.0
		for index in features:
			Val += self.w[index]
		self.nextValue = Val
	
	def Delta(self):
		self.delta = self.R - self.avgR - self.value
    
	def Delta_Update(self):
		self.delta += self.gamma*self.nextValue
    
	def Trace_Update_Critic(self,features):
		self.ev = self.gamma*self.lmbda*self.ev
		for index in features:
			self.ev[index] += 1
    
	def Trace_Update_Actor(self):
		self.e_mu = self.gamma * self.lmbda * self.e_mu + self.compatibleFeatures_mu
		self.e_sigma = self.gamma * self.lmbda * self.e_sigma + self.compatibleFeatures_sigma

	def Weights_Update_Critic(self):
		self.w += self.alphaV * self.delta * self.ev
	
	def Weights_Update_Actor(self):
		self.u_mu += self.alphaU * self.delta * self.e_mu
		self.u_sigma += self.alphaU * self.delta * self.e_sigma 
	
	def Compatible_Features(self,action,features):
		self.compatibleFeatures_mu = np.zeros(self.n)
		self.compatibleFeatures_sigma = np.zeros(self.n)
	
		# scaling factor of mean
		mcf = ((self.action - self.mean)) #/(pow(self.sigma,2))) 

		#scaling factor of sigma
		scf = (pow((self.action - self.mean),2)) - (pow(self.sigma,2)) # - 1

		if scf > 1000:
			print('scf: ' + str(scf) + ' mcf: ' + str(mcf) + '  td error: ' + str(self.delta))

		for index in features:
			self.compatibleFeatures_mu[index] = mcf
			self.compatibleFeatures_sigma[index] = scf
	    
	def Average_Reward_Update(self):
		self.avgR += self.alphaR * self.delta

	def update(self, observation, action, reward, observationNext):		
		self.R = reward
		self.Value(self.features)
		self.Delta()
		self.featuresNext = self.GetFeatures(observationNext)
		self.Next_Value(self.featuresNext)
		self.Delta_Update()
		self.Average_Reward_Update()
		self.Trace_Update_Critic(self.features)
		self.Weights_Update_Critic()
		self.Compatible_Features(action, self.features)
		self.Trace_Update_Actor()
		self.Weights_Update_Actor()

	def getAction(self, observation):
		self.features = self.GetFeatures(observation)
		self.mean = 0.0
		self.sigma = 0.1
		
		for index in self.features:
			self.mean += self.u_mu[index]
			self.sigma += self.u_sigma[index]
	
		# 	print self.sigma
		# 	print('First: ' + str(self.sigma))
		# 	#if isnan(self.sigma) or abs(self.sigma) > 10:
		# #		self.sigma = .5
			# self.sigma = exp(self.sigma)
		# 	#print('Second: ' + str(self.sigma))
		# 	if self.sigma < 1:
		# 	    self.sigma = 1	
	
		# select action with new mean
		a = np.random.normal(self.mean, 0.1)
		self.action = a

		# #a /= 100
		# if a > 1:
		#     a = 0.99
		# if a < -1:
		#     a = -0.99
		# put a in range 0-2, put a in range 0-1
		# a = ((a + 1)/2.0)*self.rangeAction + self.lowerActionBound	

		# shift normal action selection to action space
		# a = (a+1) * (self.rangeAction / 2.0) + self.lowerActionBound
 
 		a = a + 0.5

 		# round if discrete, clip to action space
		if self.isDiscrete:
			a = int(round(a))
			a = np.clip(a, self.lowerActionBound[0], self.upperActionBound[0]-1)
		
		# print(a)
		return a
	
	def Erase_Traces(self):
		self.e_mu = np.zeros(n)
		self.ev = np.zeros(n)
		self.e_sigma = np.zeros(n)

	def GetFeatures(self, observation):
		normalizedObs = [0]*self.stateDimensions
		for i in range(self.stateDimensions):
			normalizedObs[i] = (observation[i] - self.lowerStateBound[i])/self.rangeState[i]
		return self.kcoding.GetFeatures(normalizedObs)