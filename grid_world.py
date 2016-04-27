import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time
from collections import Counter

class World():
	def __init__(self,xdim,ydim,num_critters,initial_genes,mate_dif_limit):
		self.xdim = xdim
		self.ydim = ydim
		self.mate_dif_limit = mate_dif_limit
		self.vegetation = np.ones((xdim,ydim))
		#self.dead = np.zeros(xdim,ydim)
		self.crittermap = [[[] for j in xrange(ydim)] for i in xrange(xdim)]
		for i in xrange(num_critters):
			x,y = random.randint(0,xdim-1),random.randint(0,ydim-1)
			self.crittermap[x][y].append(Critter(initial_genes,0,0))

	def step(self):
		self.vegetation = np.minimum(1.0,self.vegetation+0.02)
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				to_process = self.crittermap[x][y]
				self.crittermap[x][y] = []
				while to_process:
					critter = to_process.pop()
					critter.energy -= 0.1
					move = False
					if critter.energy > 0.5:
						mate = None
						best = float('inf')
						if to_process:
							for potential_mate in to_process:
								if potential_mate.energy > 0.5:
									dif = sum([abs(gx-gy) for gx,gy in zip(critter.genes,potential_mate.genes)])
									if dif < self.mate_dif_limit:
										if dif < best:
											best = dif
											mate = potential_mate
						if mate == None:
							move = True
						else:
							child = critter.breed(mate)
							self.crittermap[x][y].append(child)
					else:
						if self.vegetation[x][y] > 0.2:
							critter.energy += 0.2
							self.vegetation[x][y] -= 0.2
						else:
							move = True
					if critter.energy >= 0.:
						if move:
							nx,ny = -1,-1
							while nx<0 or ny<0 or nx>=self.xdim or ny>=self.ydim:
								dx,dy = random.choice([(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)])
								nx,ny = x+dx,y+dy
							self.crittermap[nx][ny].append(critter)
						else:
							self.crittermap[x][y].append(critter)

	def count_critters(self):
		count = 0
		c = Counter()
		mingens = Counter()
		maxgens = Counter()
		critter_grid = np.zeros((self.xdim,self.ydim))
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				count += len(self.crittermap[x][y])
				c[len(self.crittermap[x][y])] += 1
				if self.crittermap[x][y]:
					critter_grid[x][y] = 2.
				for crit in self.crittermap[x][y]:
					mingens[crit.mingen] += 1
					maxgens[crit.maxgen] += 1
		print c
		print mingens
		print maxgens
		return count, critter_grid

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
	world = World(200,200,1000,[1],0.3)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	def update(i):
		ccount, cgrid = world.count_critters()
		print i, ccount
		ax.clear()
		ax.imshow(np.maximum(world.vegetation,cgrid),cmap='gist_earth',interpolation='nearest')
		world.step()

	a = anim.FuncAnimation(fig, update, frames=100000, repeat=False)
	plt.show()

run()
