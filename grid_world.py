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
					if critter.energy >= critter.hunger_cutoff: # critter is looking to breed
						if to_process:
							mate = to_process[0]
							dif = sum([abs(gx-gy) for gx,gy in zip(critter.genes,mate.genes)])
							if dif < self.mate_dif_limit and mate.energy >= mate.hunger_cutoff:
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

	def critter_stats(self,pops):
		genes = []
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
					genes.append(crit.genes[0])
		pops.append(count)
		print "Number of critters:", count
		print "Min generation counts:", mingens
		print "Max generation counts:", maxgens
		return critter_grid, genes

class Critter():
	def __init__(self,genes,mingen,maxgen):
		self.genes = genes
		self.energy = 0.5
		self.mingen = mingen
		self.maxgen = maxgen
		self.hunger_cutoff = genes[0]
	def breed(self,other):
		self.energy -= 0.25
		other.energy -= 0.25
		new_genes = [(g1+g2)/2-0.05+0.1*random.random() for g1,g2 in zip(self.genes,other.genes)]
		return Critter(new_genes,min(self.mingen,other.mingen)+1,max(self.maxgen,other.maxgen)+1)

def run():
	xdim = 200
	ydim = 200
	num_critters = 500
	initial_genes = [.4]
	mate_dif_limit = 0.1
	vegetation_growth_rate = 0.02
	world = World(xdim,ydim,num_critters,initial_genes,mate_dif_limit,vegetation_growth_rate)

	fig, (ax1,ax2,ax3) = plt.subplots(1,3)
	times = [-1.]*10
	pops = []

	def update(i):
		t = time.time()-start_time
		print "\nStep: %d, Time: %.2f seconds, Rate: %.2f steps/second" % (i, t, 10./(t-times[-10]))
		times.append(t)
		critter_grid, genes = world.critter_stats(pops)
		ax1.clear()
		ax1.imshow(np.maximum(world.vegetation,critter_grid),cmap='gist_earth',interpolation='nearest')
		ax2.clear()
		ax2.hist(genes,50)
		ax2.axvline(x=sum(genes)/len(genes),linewidth=4, color='r')
		ax3.clear()
		ax3.plot(range(len(pops)),pops)
		world.step()

	a = anim.FuncAnimation(fig, update, frames=100000, repeat=False)
	plt.show()

start_time = time.time()
run()
