import numpy
import math
import random
import matplotlib.pyplot as plt

class mobileRobot:
	def __init__(self):
		self.sim = plt.axes()
		plt.xlim(0,10) 
		plt.ylim(0,10)
		
		self.a = 0

		self.ans = self.a

		self.t = 0.01
		
		self.v = 2
		self.r = 0
		
		self.x = 5.0
		self.y = 5.0

		self.xns = self.x
		self.yns = self.y
			
		self.l_noise = 0.005
		self.r_noise = 0.005
		self.s_noise = 0.5

		self.lan = numpy.array([[10*random.random(), 10*random.random(), 10*random.random(), 10*random.random()],[10*random.random(), 10*random.random(), 10*random.random(), 10*random.random()]])	
		self.I = numpy.identity(3)
		self.P = numpy.array([[self.r_noise, 0, 0], [0, self.r_noise, 0], [0, 0, self.r_noise]])
		self.K = 0.5

		self.x_mat = numpy.zeros((3, 1))

	def landmarks(self):
		self.sim.scatter(self.lan[0,:], self.lan[1,:], c='r')

	def motion_dynamics(self):
		self.a = self.a + (self.r*self.t) + random.gauss(0.0, self.r_noise)
		self.x = self.x + (self.v*self.t)*math.cos(self.a) + random.gauss(0.0, self.l_noise) 
		self.y = self.y + (self.v*self.t)*math.sin(self.a) + random.gauss(0.0, self.l_noise)
		self.a %= 2*math.pi
		self.x %= 10
		self.y %= 10
		self.x_mat[0, i] = self.x
		self.x_mat[1, i] = self.y
		self.x_mat[2, i] = self.a

	def motion_dynamics_ns(self):
		self.ans = self.ans + (self.r*self.t)
		self.xns = self.xns + (self.v*self.t)*math.cos(self.ans)  
		self.yns = self.yns + (self.v*self.t)*math.sin(self.ans)
		self.ans %= 2*math.pi
		self.xns %= 10
		self.yns %= 10

	def gauss(self, mu, sigma, x):
		return math.exp(-((mu - x) ** 2) / (sigma ** 2) * 2.0) / (math.sqrt(2.0 * math.pi * (sigma ** 2)))

	def sense(self):
		z = []
		for i in range(self.lan.shape[1]):
			d = math.sqrt((self.x - self.lan[0,i])**2 + (self.y - self.lan[1,i])**2) + random.gauss(0.0, self.s_noise)
			z.append(d)
		return z	

	def meas_prob(self, z, x, y):
		prob = 1.0
		for i in range(self.lan.shape[1]):
			d = math.sqrt((x - self.lan[0,i])**2 + (y - self.lan[1,i])**2) 
			prob*=self.gauss(d, self.s_noise, z[i])
		return prob

#	def kalman_processing(self):
#		Z = self.sense()
#		y = Z - (H * self.x_mat)
#		S = H * P * H.transpose() + R
#		K = P * H.transpose() * S.inverse()
#		x = x + (K * y)
#		P = (I - (K * H)) * P
#		x = (F * x) + u
#		P = F * P * F.transpose() 

	def plot_sim(self):
		self.landmarks()
		while True:
			line = self.sim.plot([self.x - 0.25*math.cos(self.a), self.x + 0.25*math.cos(self.a)], [self.y - 0.25*math.sin(self.a), self.y + 0.25*math.sin(self.a)], c='b', linewidth=7.0) 
			line_ns = self.sim.plot([self.xns - 0.25*math.cos(self.ans), self.xns + 0.25*math.cos(self.ans)], [self.yns - 0.25*math.sin(self.ans), self.yns + 0.25*math.sin(self.ans)], c='r', linewidth=7.0)
			self.motion_dynamics_ns()			
			self.motion_dynamics()
			z = self.sense()	
			plt.pause(self.t)
			line[0].remove()
			line_ns[0].remove()
mr = mobileRobot()
mr.plot_sim()
