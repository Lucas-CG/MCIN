import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from utils import write_parameters_genetic_alg, write_report, write_performance
from optproblems import cec2005
from pathlib import Path


class Genetic_Algorithm():
	def __init__(self, function, n_genes, bounds, pop_size, n_epochs, hiper_parameters_dict, optimum=0, epsilon=10**(-6), 
		max_obj=False, show_info=False):
		self.function = function
		self.n_genes = n_genes
		self.min_value, self.max_value = bounds
		self.pop_size = pop_size
		self.n_epochs = n_epochs
		self.hiper_parameters_dict = hiper_parameters_dict
		self.optimum = optimum
		self.epsilon = epsilon
		self.max_obj = max_obj
		self.show_info = show_info
		self.count_evaluation = 0
		self.maxFes = 10000*n_genes
		self.maxFes_list = self.maxFes*(np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

		if (hiper_parameters_dict["selection_method"] == "tournment"):
			self.selection = self.tournment_selection

		else:
			print("This selection method has been not implemented")
			sys.exit()

		if (hiper_parameters_dict["cross_method"] == "blx"):
			self.crossover = self.crossover_blx

		elif (hiper_parameters_dict["cross_method"] == "aritmetic"):
			self.crossover = self.crossover_aritmetic
		else:
			print("This crossover method has been not implemented")
			sys.exit()


		if (hiper_parameters_dict["mutation_method"] == "uniform"):
			self.mutation = self.mutation_uniform

		elif(hiper_parameters_dict["mutation_method"] == "gauss"):
			self.mutation = self.mutation_gauss

		else:
			print("This crossover method has been not implemented")
			sys.exit()


	def generate_initial_population(self):
		initial_population = []
		for n in range(self.pop_size):
			cromossomo = [np.random.uniform(self.min_value, self.max_value) for i in range(self.n_genes)]
			initial_population.append(cromossomo)


		return initial_population


	def tournment_selection(self, population, apt, n=3):
		selected_pop = []
		selected_apt = []
		while len(selected_pop) < len(population):
			idx = random.sample(range(len(population)), n)
			box_population = np.array(population)[idx]
			box_apt = np.array(apt)[idx]
			idx_max = np.argmax(box_apt) if self.max_obj else np.argmin(box_apt)

			selected_pop.append(list(box_population[idx_max]))
			selected_apt.append(box_apt[idx_max])


		return selected_pop, selected_apt

	def crossover_blx(self, population, apt):
		alpha = self.hiper_parameters_dict["cross_alpha"]
		cross_ratio = self.hiper_parameters_dict["cross_ratio"]
		crossover_pop = []

		while len(crossover_pop) < len(population):

			parents_idx = random.sample(range(len(population)), 2)
			parent1, parent2 = population[parents_idx[0]], population[parents_idx[1]]
			apt_parent1, apt_parent2 = apt[parents_idx[0]], apt[parents_idx[1]]

			r = random.random()

			if (r < cross_ratio):
				beta = np.random.uniform(-alpha, 1+alpha)
				children = np.array(parent1) + beta*(np.array(parent1)-np.array(parent2))
				
				if (np.all(children) > self.min_value and np.all(children) > self.max_value):
					crossover_pop.append(children)

			else:
				check = apt_parent1 > apt_parent2 if self.max_obj else apt_parent1 < apt_parent2 
				if (check):
					crossover_pop.append(parent1)
				else:
					crossover_pop.append(parent2)

		return crossover_pop


	def crossover_aritmetic(self, population, apt):
		cross_ratio = self.hiper_parameters_dict["cross_ratio"]
		crossover_pop = []

		while len(crossover_pop) < len(population):

			parents_idx = random.sample(range(len(population)), 2)
			parent1, parent2 = population[parents_idx[0]], population[parents_idx[1]]
			apt_parent1, apt_parent2 = apt[parents_idx[0]], apt[parents_idx[1]]

			r = random.random()
			if(r < cross_ratio):
				beta = np.random.uniform(0, 1)
				children1 = beta*np.array(parent1) + (1-beta)*np.array(parent2)
				children2 = beta*np.array(parent2) + (1-beta)*np.array(parent1)
				if (self.function([children1]) < self.function([children2])):
					children = children1
				else:
					children = children2

				#if ((self.min_value < np.all(children) < self.max_value)):
				#	crossover_pop.append(children)
				crossover_pop.append(np.clip(children, self.min_value, self.max_value))
			else:
				if(apt_parent1 >= apt_parent2):
					crossover_pop.append(parent1)
				else:
					crossover_pop.append(parent2)

		return crossover_pop

	def mutation_uniform(self, population,epoch):
		mutation_ratio = self.hiper_parameters_dict["mutation_ratio"]
		for cromo in population:
			for i in range(len(cromo)):
				r = random.random()
				if(r < mutation_ratio):
					cromo[i] = np.random.uniform(self.min_value, self.max_value)

		return population

	def mutation_gauss(self, population, epoch):
		mutated_pop = []
		mutation_rate = self.hiper_parameters_dict["mutation_ratio"]

		for cromo in population:
			mutated_ind = []
			for ind in cromo:
				r = random.random()
				if (r < mutation_rate):
					sigma = 1 - 0.9*(epoch/self.n_epochs)
					mutated = ind+np.random.normal(0, sigma)
					mutated = np.clip(mutated, self.min_value, self.max_value)
					#if (mutated > self.max_value):
					#	mutated = self.max_value

					#if (mutated < self.min_value):
					#	mutated = self.min_value

					mutated_ind.append(mutated)

				else:
					mutated_ind.append(ind)

			mutated_pop.append(mutated_ind)
		return mutated_pop



	def ordering(self, population, apt):
		sorted_apt = sorted(apt, reverse=True) if self.max_obj else sorted(apt, reverse=False)
		sorted_pop = []
		for idx, apt_ind  in enumerate(sorted_apt):
			sorted_pop.append(population[list(apt).index(apt_ind)])


		return sorted_pop, sorted_apt

	def run(self):
		population = self.generate_initial_population()
		apt = self.function(population)
		self.count_evaluation+=1
		epoch = 0
		best_apt = np.max(apt) if self.max_obj else np.min(apt)
		error = abs(best_apt - self.optimum)
		error_evolution = []
		
		while (epoch < self.n_epochs and error > self.epsilon):

			best_cromo = population[np.argmin(apt)]
			best_current_apt = np.min(apt)

			population, apt = self.selection(population, apt)
			population = self.crossover(population, apt)
			population = self.mutation(population, epoch)
			population.append(best_cromo)

			apt = self.function(population)
			apt.append(best_current_apt)
			

			best_apt = np.min(apt)
			error = abs(best_apt - self.optimum)
			epoch += 1

			if (self.show_info):
				print("Epoch: %s, Best Fitness: %s, Best Error: %s"%(epoch, best_apt, error))


			if(self.count_evaluation in self.maxFes_list):
				error_evolution.append(error)
		
		print("Epoch: %s, Best Fitness: %s, Best Error: %s"%(epoch, best_apt, error))
		success = 1 if error < self.epsilon else 0
		return error, success, error_evolution, epoch

