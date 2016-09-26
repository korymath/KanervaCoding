import random
import math
import operator
import numpy as np
import struct

class KanervaCoder:
	distanceMeasure = 'euclidian'
	numPrototypes = 50
	dimensions = 1
	threshold = 0.02
	numClosest = 10
	prototypes = None
	visitCounts = None
	updatePrototypes = None
	minNumberVisited = 50

	updateFunc = None

	activationRadii = 0
	beenAroundTheBlock = False # set to true once a single prototype has been visited the minNumberVisited

	def __init__(self, _numPrototypes, _dimensions):

		self.dimensions = _dimensions
		self.numPrototypes = _numPrototypes
		self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
		self.visitCounts = np.zeros(self.numPrototypes)
		self.updatedPrototypes = []
		self.minNumberVisited = 1000000
		self.updateFunc = 0
		self.activationRadii = .1


	def GetFeatures(self, data):
		if self.distanceMeasure == 'euclidian':
			#tempArr = np.array([1 if np.linalg.norm(data - self.prototypes[i]) < self.threshold else 0 for i in range(len(self.prototypes))])
			
			if self.updateFunc == 0: # XGame Paper

				tempArr = np.array([[i, np.linalg.norm(data - self.prototypes[i])] for i in range(len(self.prototypes))])

				closestPrototypesIndxs = [int(x[0]) for x in sorted(tempArr, key = lambda x: x[1])[:self.numClosest]]
				
				for i in closestPrototypesIndxs:
					self.visitCounts[i] += 1

				if self.beenAroundTheBlock == False: # use this so we dont have to calculated the max every time
					maxVisit = max(self.visitCounts)
					if maxVisit > self.minNumberVisited:
						print('Updating')
						self.beenAroundTheBlock = True
						
				if self.beenAroundTheBlock:
					self.updatePrototypesXGame()

			elif self.updateFunc == 1: # Case Studies

				closestPrototypesIndxs = []
				data = np.array(data)
				for prototype in range(self.numPrototypes):
					diffArr = abs(data - self.prototypes[prototype])
					#closestPrototypesIndxs.append(min([1 - diff/self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr]))
					u = min([1 - diff/self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr])
					if u > 0:
						closestPrototypesIndxs.append(prototype)
				



			return closestPrototypesIndxs


		else:
			# fuzzy
			#return np.array([self.computeHamming(data,i)/self.threshold for i in range(len(self.prototypes))])
			tempArr =  np.array([1 if self.computeHamming(data,i) < self.threshold else 0 for i in range(len(self.prototypes))])

			return np.where(tempArr == 1)[0]



	def updatePrototypesXGame(self):
		self.updatedPrototypes = []
		mostVisitedPrototypeIndexs = [i[0] for i in sorted(enumerate(self.visitCounts), key=lambda x:x[1])]
		count = 0
		for prototype in range(self.numPrototypes):
			if math.exp(-self.visitCounts[prototype]) > random.random(): # remove with probability e^-m (Equation 4)
				self.visitCounts[prototype] = 0
				replacementPrototypeIndex = mostVisitedPrototypeIndexs[-(count+1)]
				self.prototypes[prototype] = self.prototypes[replacementPrototypeIndex] # add another prototype
				

				for dimension in range(self.dimensions):
					randOffset = (random.random() - .5)/(self.numPrototypes^-self.dimensions)
					self.prototypes[prototype][dimension] += randOffset # change every dimension to something close by
				

				self.updatedPrototypes.append([prototype, self.prototypes[prototype], replacementPrototypeIndex])
				count += 1

		self.visitCounts = np.zeros(self.numPrototypes)
		self.beenAroundTheBlock = False			

		print('Done updatedPrototypes')