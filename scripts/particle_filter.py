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
		
		self.v = 5
		self.r = 0
		
		self.x = 5.0
		self.y = 5.0

		self.xns = self.x
		self.yns = self.y
			
		self.num_part = 10000	
		self.p = numpy.zeros((3, self.num_part))
		self.p1 = numpy.zeros((3, self.num_part))

		self.l_noise = 0.005
		self.r_noise = 0.005
		self.s_noise = 0.5

		self.num_lan = 10
		self.lan = numpy.zeros((2, self.num_lan))
		for i in range(self.num_lan):
			self.lan[0, i] = 10 * random.random()	
			self.lan[1, i] = 10 * random.random()	
		self.w = []

	def gen_particles(self):
		for i in range(self.num_part):
			self.p[0, i] = 10 * random.random()
			self.p[1, i] = 10 * random.random()
			self.p[2, i] = random.randint(-314, 315) / 100	

	def landmarks(self):
		self.sim.scatter(self.lan[0,:], self.lan[1,:], c='r')

	def motion_dynamics(self):
		self.a = self.a + (self.r*self.t) + random.gauss(0.0, self.r_noise)
		self.x = self.x + (self.v*self.t)*math.cos(self.a) + random.gauss(0.0, self.l_noise) 
		self.y = self.y + (self.v*self.t)*math.sin(self.a) + random.gauss(0.0, self.l_noise)
		self.a %= 2*math.pi
		self.x %= 10
		self.y %= 10
	
	def motion_dynamics_ns(self):
		self.ans = self.ans + (self.r*self.t)
		self.xns = self.xns + (self.v*self.t)*math.cos(self.ans)  
		self.yns = self.yns + (self.v*self.t)*math.sin(self.ans)
		self.ans %= 2*math.pi
		self.xns %= 10
		self.yns %= 10

	def move_particles(self):
		for i in range(self.num_part):
			self.p[2, i] = self.p[2, i] + (self.r*self.t)
			self.p[0, i] = self.p[0, i] + (self.v*self.t)*math.cos(self.p[2, i]) 
			self.p[1, i] = self.p[1, i] + (self.v*self.t)*math.sin(self.p[2, i])
			self.p[2, i] %= 2*math.pi
			self.p[0, i] %= 10
			self.p[1, i] %= 10

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

	def plot_sim(self):
		self.landmarks()
		self.gen_particles()
		while True:
			part = self.sim.scatter(self.p[0,:], self.p[1,:], c='g')
			line = self.sim.plot([self.x - 0.25*math.cos(self.a), self.x + 0.25*math.cos(self.a)], [self.y - 0.25*math.sin(self.a), self.y + 0.25*math.sin(self.a)], c='b', linewidth=7.0) 
			line_ns = self.sim.plot([self.xns - 0.25*math.cos(self.ans), self.xns + 0.25*math.cos(self.ans)], [self.yns - 0.25*math.sin(self.ans), self.yns + 0.25*math.sin(self.ans)], c='r', linewidth=7.0)
			self.motion_dynamics_ns()			
			self.motion_dynamics()
			z = self.sense()	
			self.move_particles()
			w_sum = 0
			self.w = []
			for i in range(self.num_part):
				self.w.append(self.meas_prob(z, self.p[0, i], self.p[1, i]))
				w_sum = w_sum + self.w[i]			
			alpha = []
			for i in range(self.num_part):
				alpha.append(self.w[i]/w_sum)
			a_np = numpy.array(alpha)
			ind = numpy.argsort(a_np)
			alpha.sort()
			for i in range(1, self.num_part):
				alpha[i] = alpha[i] + alpha[i-1]
			new_p = []
			j = 0 
			while j < self.num_part:
				pull = random.random()
				for i in range(self.num_part):
					if pull <= alpha[i]:
						new_p.append(ind[i])
						j = j + 1
						break
		        for i in range(self.num_part):
				self.p1[0, i] = self.p[0, new_p[i]] 
				self.p1[1, i] = self.p[1, new_p[i]] 
				self.p1[2, i] = self.p[2, new_p[i]] 	
			self.p = self.p1
			plt.pause(self.t)
			line[0].remove()
			line_ns[0].remove()
			part.remove()
#			x = sum(self.p[0, :].tolist())/self.p.shape[1]
#			y = sum(self.p[1, :].tolist())/self.p.shape[1]
#			print('x:' + str(x))
#			print('y:' + str(y))
mr = mobileRobot()
mr.plot_sim()
