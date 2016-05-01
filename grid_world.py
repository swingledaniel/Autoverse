import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time
from collections import Counter

class World():
	def __init__(self,xdim,ydim,num_critters,initial_genes,mate_dif_limit, cannib_dif_limit, vegetation_growth_rate):
		self.num_genes = len(initial_genes)
		self.xdim = xdim
		self.ydim = ydim
		self.vegetation = 0.3*np.ones((xdim,ydim))
		self.vegetation_growth_rate = vegetation_growth_rate
		#self.dead = np.zeros(xdim,ydim)
		self.crittermap = [[[] for j in xrange(ydim)] for i in xrange(xdim)]
		for i in xrange(num_critters):
			x,y = random.randint(0,xdim-1),random.randint(0,ydim-1)
			self.crittermap[x][y].append(Critter(initial_genes,initial_genes[1]*2,0,0,mate_dif_limit, cannib_dif_limit))
		self.stats_births = 0
		self.stats_starved = 0
		self.stats_eaten = 0

	def step(self):
		self.vegetation = np.minimum(1.0,self.vegetation+self.vegetation_growth_rate)
		for x in xrange(self.xdim):
			for y in xrange(self.ydim):
				to_process = self.crittermap[x][y]
				self.crittermap[x][y] = []
				while to_process:
					critter = to_process.pop()

					veg_level = self.vegetation[x][y]
					nearby_critters = to_process + self.crittermap[x][y]
					action, child, veg_diff = critter.step(veg_level,nearby_critters)
					
					if action == 0:
						if critter.energy <= -100.0:
							self.stats_eaten += 1
						else:
							self.stats_starved += 1

					# action == 0: creature dies
					# action == 1: stays in place
					# action == 2: moves to random nearby location

					if action == 1: # stays in place
						self.crittermap[x][y].append(critter)
					elif action == 2: # moves to random nearby location
						dx,dy = random.choice([(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)])
						nx,ny = (x+dx)%self.xdim, (y+dy)%self.ydim # borders loop around
						self.crittermap[nx][ny].append(critter)

					if child is not None:
						self.stats_births += 1
						dx,dy = random.choice([(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)])
						nx,ny = (x+dx)%self.xdim, (y+dy)%self.ydim # borders loop around
						self.crittermap[nx][ny].append(child)

					self.vegetation[x][y] += veg_diff

	def critter_stats(self,pops):
		genes = [[] for i in xrange(self.num_genes)]
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
					for i in xrange(self.num_genes):
						genes[i].append(crit.genes[i])
		pops.append(count)
		print "Number of critters:", count
		print "Number of Births:", self.stats_births
		print "Number of Starves:", self.stats_starved
		print "Number of Eaten:", self.stats_eaten
		self.stats_births = 0
		self.stats_starved = 0
		self.stats_eaten = 0
		#print "Min generation counts:", mingens
		#print "Max generation counts:", maxgens
		return critter_grid, genes

class Critter():
	def __init__(self,genes,init_energy,mingen,maxgen,mate_dif_limit, cannib_dif_limit):
		self.age = 0.
		self.stage = 0
		self.genes = genes
		self.energy = init_energy
		self.mingen = mingen
		self.maxgen = maxgen
		self.mate_dif_limit = mate_dif_limit
		self.cannib_dif_limit = cannib_dif_limit
		self.hunger_cutoff = genes[0] # how much energy creature needs to consider breeding
		self.breeding_cutoff = genes[1] # how much energy creature needs to only try to breed, not eat
		self.energy_contributed_to_offspring = genes[2]
		self.veg_digestion_rate = genes[3]
		self.meat_digestion_rate = genes[4]

	def breed(self,other):
		self.energy -= self.energy_contributed_to_offspring
		other.energy -= other.energy_contributed_to_offspring
		new_energy = self.energy_contributed_to_offspring + other.energy_contributed_to_offspring
		new_genes = [random.choice([g1,g2,(g1+g2)/2]) for g1,g2 in zip(self.genes,other.genes)]
		new_genes = [max(0,random.choice([g-0.1+0.2*random.random(),g*(.9+0.2*random.random())])) for g in new_genes]
		return Critter(new_genes,new_energy,min(self.mingen,other.mingen)+1,max(self.maxgen,other.maxgen)+1,self.mate_dif_limit, self.cannib_dif_limit)

	def step(self,veg_level,nearby_critters):
		if self.energy <= 0.:
			return 0, None, 0. # creature dies

		self.energy -= 0.02 + 0.00003*self.age + self.veg_digestion_rate/50. + self.meat_digestion_rate/50. # energy drains
		self.age += 1

		if self.energy >= self.hunger_cutoff: # has enough energy to breed
			if self.stage == 2:
				if nearby_critters:
					pot_mate = nearby_critters[0]
					if pot_mate.energy >= pot_mate.hunger_cutoff:
						dif = sum([abs(gx-gy) for gx,gy in zip(self.genes,pot_mate.genes)])
						if dif < self.mate_dif_limit:
							child = self.breed(pot_mate)
							return 2, child, 0.

		if self.energy < self.breeding_cutoff: # doesn't have enough energy to justify not eating
			pot_veg = 0.
			pot_meat = 0.
			if veg_level >= 0.2:
				pot_veg = 0.2*(1-1./(1+self.veg_digestion_rate))
				#self.energy += 0.2*self.veg_digestion_rate
				#return 1, None, -0.2
			if nearby_critters:
				pot_prey = nearby_critters[0]

				#check if potential prey is genetically different enough
				#if dif is too low, meat value is 0, elif energy less than own then meat formula
				dif = sum([abs(gx-gy) for gx,gy in zip(self.genes,pot_prey.genes)])
				if dif < self.cannib_dif_limit:
					pot_meat = 0
				elif pot_prey.energy + pot_prey.stage*pot_prey.breeding_cutoff < self.energy + self.stage*self.breeding_cutoff:
					pot_meat = pot_prey.energy*self.meat_digestion_rate
			if max(pot_veg,pot_meat) > 0.:
				if pot_veg > pot_meat:
					self.energy += pot_veg
					return 1, None, -0.2
				else:
					self.energy += pot_meat
					pot_prey.energy = -100.0
					return 1, None, 0.
		elif self.stage < 2:
			self.stage += 1
			self.energy = self.hunger_cutoff
			#self.energy /= 2
					
		return 2, None, 0. # couldn't do what it wanted, so moves

def run(steps_per_redraw=1):
	xdim = 200
	ydim = 200
	num_critters = 100
	initial_genes = [.5,0.75,0.25,1.,.0]
	num_genes = len(initial_genes)
	mate_dif_limit = .3
	cannib_dif_limit = .3
	vegetation_growth_rate = 0.02
	world = World(xdim,ydim,num_critters,initial_genes,mate_dif_limit, cannib_dif_limit, vegetation_growth_rate)

	fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
	times = [-1.]*10
	pops = []
	gene_aves = [[] for i in xrange(num_genes)]

	#gene legend dictionary for holding names
	names = {
	0 : 'Breedmin',
	1 : 'Feedmax',
	2 : 'Offspring energy contrib',
	3 : 'Veglovin',
	4 : 'Meatlovin',
	5 : 'Cannibmax'
	}

	def update(i):
		t = time.time()-start_time
		print "\nStep: %d, Time: %.2f seconds, Rate: %.2f steps/second" % (i*steps_per_redraw, t, 10.*steps_per_redraw/(t-times[-10]))
		times.append(t)
		critter_grid, genes = world.critter_stats(pops)
		ax1.clear()
		ax1.imshow(np.maximum(world.vegetation,critter_grid),cmap='gist_earth',interpolation='nearest')
		ax2.clear()
		for i in xrange(num_genes):
			ax2.hist(genes[i],10)
		gene_ave = [sum(genes[i])/len(genes[i]) for i in xrange(num_genes)]
		for i in xrange(num_genes):
			gene_aves[i].append(gene_ave[i])
		#ax2.axvline(x=gene_ave,linewidth=4, color='r')
		ax3.clear()
		ax3.plot(range(len(pops)),pops)
		ax3.set_title('Population')
		ax4.clear()
		for i in xrange(num_genes):
			ax4.plot(range(len(gene_aves[i])),gene_aves[i], label=names[i])
		ax4.legend(loc='best').get_frame().set_alpha(0.5)
		ax4.set_title('Gene Average')

		for i in xrange(steps_per_redraw):
			world.step()

	start_time = time.time()
	a = anim.FuncAnimation(fig, update, frames=100000, repeat=False)
	plt.show()

run(100)