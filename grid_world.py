import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time
from collections import Counter

class World():
	def __init__(self,xdim,ydim,num_critters,initial_genes,mate_dif_limit,vegetation_growth_rate):
		self.xdim = xdim
		self.ydim = ydim
		self.mate_dif_limit = mate_dif_limit
		self.vegetation = 0.5*np.ones((xdim,ydim))
		self.vegetation_growth_rate = vegetation_growth_rate
		#self.dead = np.zeros(xdim,ydim)
		self.crittermap = [[[] for j in xrange(ydim)] for i in xrange(xdim)]
		for i in xrange(num_critters):
			x,y = random.randint(0,xdim-1),random.randint(0,ydim-1)
			self.crittermap[x][y].append(Critter(initial_genes,0,0))

	def step(self):
		self.vegetation = np.minimum(1.0,self.vegetation+self.vegetation_growth_rate)
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				to_process = self.crittermap[x][y]
				self.crittermap[x][y] = []
				while to_process:
					critter = to_process.pop()
					if critter.energy <= 0.:
						continue # kills critter
					critter.energy -= 0.1
					move = False
					if critter.energy > 0.5: # critter is looking to breed
						if to_process:
							mate = to_process[0]
							dif = sum([abs(gx-gy) for gx,gy in zip(critter.genes,mate.genes)])
							if dif < self.mate_dif_limit:
								child = critter.breed(mate)
								self.crittermap[x][y].append(child)
						else:
							move = True
					else: # critter is looking to eat
						if self.vegetation[x][y] > 0.2:
							critter.energy += 0.2
							self.vegetation[x][y] -= 0.2
						else:
							move = True
					if move:
						dx,dy = random.choice([(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)])
						nx,ny = (x+dx)%self.xdim, (y+dy)%self.ydim
						self.crittermap[nx][ny].append(critter)
					else:
						self.crittermap[x][y].append(critter)

	def critter_stats(self):
		count = 0
		mingens = Counter()
		maxgens = Counter()
		critter_grid = np.zeros((self.xdim,self.ydim))
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				count += len(self.crittermap[x][y])
				if self.crittermap[x][y]:
					critter_grid[x][y] = 2.
				for crit in self.crittermap[x][y]:
					mingens[crit.mingen] += 1
					maxgens[crit.maxgen] += 1
		print "Number of critters:", count
		print "Min generation counts:", mingens
		print "Max generation counts:", maxgens
		return critter_grid

class Critter():
	def __init__(self,genes,mingen,maxgen):
		self.genes = genes
		self.energy = 0.5
		self.mingen = mingen
		self.maxgen = maxgen
	def breed(self,other):
		self.energy -= 0.25
		other.energy -= 0.25
		return Critter(self.genes,min(self.mingen,other.mingen)+1,max(self.maxgen,other.maxgen)+1)

def run():
	xdim = 200
	ydim = 200
	num_critters = 5000
	initial_genes = [1]
	mate_dif_limit = 0.3
	vegetation_growth_rate = 0.02
	world = World(xdim,ydim,num_critters,initial_genes,mate_dif_limit,vegetation_growth_rate)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	def update(i):
		print "\nStep: %d Time: %.2f seconds" % (i, time.time() - start_time)
		critter_grid = world.critter_stats()
		ax.clear()
		ax.imshow(np.maximum(world.vegetation,critter_grid),cmap='gist_earth',interpolation='nearest')
		world.step()

	a = anim.FuncAnimation(fig, update, frames=1000, repeat=False)
	plt.show()

start_time = time.time()
run()
