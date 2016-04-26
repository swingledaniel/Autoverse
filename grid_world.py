import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time

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
			self.crittermap[x][y].append(Critter(initial_genes))

	def step(self):
		self.vegetation = np.minimum(1.0,self.vegetation+0.01)
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
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				count += len(self.crittermap[x][y])
		return count

class Critter():
	def __init__(self,genes):
		self.genes = genes
		self.energy = 0.5
	def breed(self,other):
		self.energy -= 0.25
		other.energy -= 0.25
		return Critter(self.genes)

'''
def run():
	print "hello"
	world = World(20,20,10,[1],0.3)

	fig = plt.figure()

	for i in xrange(100):
		print i, world.count_critters()
		plt.clf()
		plt.imshow(world.vegetation)
		fig.canvas.draw()
		time.sleep(1.0)
		world.step()
'''

def run():
	world = World(50,50,20,[1],0.3)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	def update(i):
		print i, world.count_critters()
		ax.clear()
		ax.imshow(world.vegetation,cmap='Greens',interpolation='nearest')
		world.step()

	a = anim.FuncAnimation(fig, update, frames=100000, repeat=False)
	plt.show()

run()
