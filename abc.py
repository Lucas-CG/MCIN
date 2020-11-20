import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
import pandas as pd
from optproblems import cec2005
from pathlib import Path
from copy import deepcopy


class Function_Objective(object):
	def __init__(self, dim, bounds, max_obj=False):
		self.dim = dim
		self.min_value, self.max_value = bounds
		self.max_obj = max_obj
		self.count_fes = 0

	#generate random position
	def generate_random_position(self):
		return np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim)

	def evaluate(self, x):
		pass


class F1(Function_Objective):
	def __init__(self, dim):
		super(F1, self).__init__(dim, [-100.0, 100.0], max_obj=False)
		self.dim = dim

	def evaluate(self, x):
		f1 = cec2005.F5(self.dim)
		self.count_fes+=1
		return f1(x)

# REMEMBER TO INSERT A FES COUNTER
class Bee(object):
	#initialize a bee
	"""A Bee requires three main tasks"""
	def __init__(self, function):
		self.function = function
		self.min_value, self.max_value = function.min_value, function.max_value
		self.dim = function.dim
		self.max_obj = function.max_obj
		self.xi = function.generate_random_position()
		self.fitness = function.evaluate(self.xi)
		self.trial = 0
		self.prob = 0

	#evaluate if a position belongs to the boundary space
	def evaluate_decision_boundary(self, current_pos):
		return np.clip(current_pos, self.min_value, self.max_value)

	# updates the current position, if current fitness is better than the old fitness
	def update_bee(self, pos, fitness):
		check_update = fitness>self.fitness if self.max_obj else fitness < self.fitness
		if (check_update):
			self.fitness = fitness
			self.xi = pos
			self.trial = 0
		else:
			self.trial+=1

	# when food source is abandoned (e.g.; self.trial > MAX), this generates a random food source e send be to there.  
	def reset_bee(self, max_trial):
		if (self.trial > max_trial):
			self.xi = self.function.generate_random_position()
			self.fitness = self.function.evaluate(self.xi)
			self.trial = 0

 
class EmployeeBee(Bee):
	def explore(self, max_trial, bee_idx, swarm):
		idxs = [idx for idx in range(len(swarm)) if idx!=bee_idx]
		if (self.trial <= max_trial):
			phi = np.random.uniform(low=-1, high=1, size=self.dim)
			other_bee = swarm[random.choice(idxs)]
			new_xi = self.xi + phi*(self.xi - other_bee.xi)
			new_xi = self.evaluate_decision_boundary(new_xi)
			new_fitness = self.function.evaluate(new_xi)
			self.update_bee(new_xi, new_fitness)
		else:
			self.reset_bee(max_trial)


	def get_fitness(self):
		return 1/(1+self.fitness) if self.fitness >= 0 else 1+abs(self.fitness)

	def compute_probability(self, max_fitness):
		self.prob = self.get_fitness()/max_fitness

class OnlookBee(Bee):
	def onlook(self, best_food_sources, max_trials):
		candidate = np.random.choice(best_food_sources)
		self.exploit(candidate.xi, candidate.fitness, max_trials)

	def exploit(self, candidate, fitness, max_trials):
		if (self.trial <= max_trials):
			component = np.random.choice(candidate)
			phi = np.random.uniform(low=-1, high=1, size=len(candidate))
			n_pos = candidate + phi*(candidate - component)
			n_pos = self.evaluate_decision_boundary(n_pos)
			n_fitness = self.function.evaluate(n_pos)
			check_update = n_fitness > self.fitness if self.max_obj else n_fitness < self.fitness
			if (check_update):
				self.fitness = n_fitness
				self.xi = n_pos
				self.trial = 0
			else:
				self.trial+=1

class ABC(object):
	def __init__(self, function, colony_size, dim, optimum, maxFes=10000, max_trials=100, max_obj=False, epsilon=10**(-8)):
		self.function = function
		self.colony_size = colony_size
		self.max_trials = max_trials
		self.epsilon = epsilon
		self.dim = dim
		self.maxFes = dim*maxFes
		self.optimal_solution = None
		self.optimality_tracking = []
		self.optimum = optimum
		self.max_obj = max_obj

	def reset_algorithm(self):
		self.optimal_solution = None
		self.optimality_tracking = []

	

	def update_optimality_tracking(self):
		self.optimality_tracking.append(self.optimal_solution)

	def initialize_employees(self):
		self.employee_bees = [EmployeeBee(self.function) for idx in range(self.colony_size // 2)]

	def update_optimal_solution(self):
		#print(min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness))
		swarm_fitness_list = []
		for bee in (self.onlookers_bees + self.employee_bees):
			swarm_fitness_list.append(bee.fitness)

		n_optimal_solution = max(swarm_fitness_list) if self.max_obj else min(swarm_fitness_list)
		#n_optimal_solution = min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness)
		if not self.optimal_solution:
			self.optimal_solution = deepcopy(n_optimal_solution)
		else:
			if n_optimal_solution < self.optimal_solution:
				self.optimal_solution = deepcopy(n_optimal_solution)


	def initialize_onlookers(self):
		self.onlookers_bees = [OnlookBee(self.function) for idx in range(self.colony_size // 2)]


	def employee_bee_phase(self):
		for i, bee in enumerate(self.employee_bees):
			bee.explore(self.max_trials, i, self.employee_bees)
		#map(lambda idx, bee: bee.explore(self.max_trials, idx, self.employee_bees), self.employee_bees)

	def calculate_probabilities(self):
		sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
		#map(lambda bee: bee.compute_probability(sum_fitness), self.employee_bees)
		for bee in self.employee_bees:
			bee.compute_probability(sum_fitness)

	def select_best_food_sources(self):

		self.best_food_sources = []
		while (len(self.best_food_sources))==0:
			self.best_food_sources = [bee for bee in self.employee_bees if bee.prob > np.random.uniform(0,1)]
		
		#self.best_food_sources =\
		# filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)

		#print(list(self.best_food_sources), len(list(self.best_food_sources)))
		#while len(list(self.best_food_sources))==0:
		#	print("oi")
		#	self.best_food_sources =\
		#	 filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)
		#print(list(self.best_food_sources), len(list(self.best_food_sources)))

		#sys.exit()

	def onlookers_bee_phase(self):
		for bee in self.onlookers_bees:
			bee.onlook(self.best_food_sources, self.max_trials)
		
		#map(lambda idx, bee: bee.onlook(self.best_food_sources, self.max_trials), self.onlookers_bees)

	def scout_bee_phase(self):
		map(lambda bee: bee.reset_bee(self.max_trials), self.onlookers_bees + self.employee_bees)


	def optimize(self, show_info=True):
		self.reset_algorithm()
		self.initialize_employees()
		self.initialize_onlookers()
		best_error = np.inf
		epoch = 0
		while (self.function.count_fes < self.maxFes and (best_error > self.epsilon)):
			self.employee_bee_phase()
			self.update_optimal_solution()

			self.calculate_probabilities()
			self.select_best_food_sources()

			self.onlookers_bee_phase()
			self.scout_bee_phase()

			self.update_optimal_solution()
			self.update_optimality_tracking()

			best_error = abs(self.optimum - self.optimal_solution)
			if(show_info):
				print("Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))
			epoch+=1
		
		print("FINAL: Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))

n_epoch = 100000
dim = 10
f1 = F1(dim)
colony_size = 20
optimum = -310
max_trials = 100
bee_colony = ABC(f1, colony_size, dim, optimum, max_trials=300)
bee_colony.optimize()