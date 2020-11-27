import random
import math
import numpy as np, os, sys
from optproblems import cec2005
from pathlib import Path
from copy import deepcopy
from utils import write_parameters_acor, write_performance, write_evolution


class ACOR(object):
	def __init__(self, function, popSize, dim, bounds, parameters, optimum=-450, maxFes=10000, max_obj=False, epsilon=10**(-8)):
		self.function = function
		self.popSize = popSize
		self.min_value, self.max_value = bounds
		self.dim = dim
		self.optimum = optimum
		self.max_obj = max_obj
		self.epsilon = epsilon
		self.pressure = parameters["selection_pressure"]
		self.archive_size = parameters["archive_size"]
		self.evaporation_rate = parameters["evaporation_rate"]
		self.nrSample = parameters["nrSample"]
		self.collect_error_rate = 100
		self.count_fes = 0
		self.best_error = np.inf
		self.error_evolution_list = []

		self.maxFes = dim*maxFes
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

	
	def evaluate(self, X):
		z = []
		for ant in X:
			z.append(self.function(ant))
			self.count_fes+=1
		
		return np.array(z)

	def evaluate_ind(self, x):
		z = self.function(x)
		self.count_fes+=1

		return z

	
	def compute_weights(self):
		numerator = np.exp((np.power(np.arange(1, self.archive_size+1), 2))/(2*(self.pressure**2)*(self.archive_size**2)))
		den = np.sqrt(2*np.pi)*self.pressure*self.archive_size
		return numerator/den

	def ordering(self, pop, fit):
		sorted_fit = np.sort(fit)
		sorted_pop = pop[np.argsort(fit)]
		return sorted_pop, sorted_fit

	def roullete_wheel(self, pop, fit, p):
		selected_idx = []
		cum_prob = 0
		r = np.random.uniform(0, 1)
		for i in range(self.archive_size):
			cum_prob += p[i]
			if (r < cum_prob):
				selected_idx.append(i)

		return pop[selected_idx], fit[selected_idx] 




	def run(self, show_info=True):
		epoch = 0
		pop = np.random.uniform(self.min_value, self.max_value, size=(self.popSize, self.dim))
		fitness = self.evaluate(pop)
		pop, fit = self.ordering(pop, fitness)
		best_ant = pop[0]
		best_fit = fit[0]
		best_error = abs(best_fit-self.optimum)
		weights = self.compute_weights()
		prob_selection = weights/np.sum(weights)
		best_error_miss = True
		fes_final = None		

		while (self.count_fes < self.maxFes and best_error>self.epsilon):
			if(show_info):
				print("Epoch: %s, Fes: %s, Best Error: %s"%(epoch, self.count_fes, best_error))
				if(self.count_fes%self.collect_error_rate==0):
					self.error_evolution_list.append(best_error)


			solutions = pop

			sigma = np.zeros((self.archive_size, self.dim))
			for i in range(self.archive_size):
				D = 0
				for r in range(self.archive_size):
					D += abs(solutions[i] - solutions[r]) # maybe, add a log
				
				sigma[i] = (self.evaporation_rate*D)/(self.archive_size+1)


			new_pop = np.zeros((self.nrSample, self.dim))
			new_fit = np.zeros(self.nrSample)
			for i in range(self.nrSample):
				#for j in range(self.dim): 
				selected_solution = solutions[np.random.choice(self.popSize, p=prob_selection)]
				gausian_avg = selected_solution
				new_pop[i] = np.random.normal(gausian_avg, sigma[i])
				new_fit[i] = self.evaluate_ind(np.clip(new_pop[i], self.min_value, self.max_value))	

			conc_pop = np.concatenate((pop, new_pop))
			conc_fit = np.concatenate((fit, new_fit))
			sorted_pop, sorted_fit = self.ordering(conc_pop, conc_fit)
			pop, fit = sorted_pop[:self.archive_size], sorted_fit[:self.archive_size]

			best_fit = sorted_fit[0]
			best_error = abs(best_fit-self.optimum)

			if (best_error < self.epsilon and best_error_miss):
				best_error_final = best_error
				best_error_miss = False
				fes_final = self.count_fes				

			if (best_error_miss):
				fes_final = self.count_fes
				best_error_final = best_error

			epoch+=1

		print("FINAL: Epoch: %s, FES: %s, Best Error: %s"%(epoch, fes_final, best_error_final))
		success = 1 if best_error_final < self.epsilon else 0
		return best_error_final, success, self.error_evolution_list, fes_final
		


if __name__ == '__main__':


	dim = 10
	popSize = 30
	bounds = [-100, 100]
	optimum = -310
	selection_pressure = .9
	archive_size = popSize
	evaporation_rate = 0.7
	nrSample = 10
	parameters = {"selection_pressure": selection_pressure, "archive_size":archive_size, 
	"evaporation_rate": evaporation_rate, "nrSample": nrSample}
	function_name = "F5"
	savePathParameters = Path("plots/acor_parameters.csv")
	savePathReporter = Path("plots/acor_statistics_10.csv")
	savePathEvolution = Path("plots/%s_acor_evolution_error.csv"%(function_name))


	f1 = cec2005.F5(dim)
	n_runs = 25
	error_list = []
	error_evolution_list = []
	fes_list = []
	success_count = 0
	for i in range(n_runs):
		ant_colony = ACOR(f1, popSize, dim, bounds, parameters, optimum=optimum)
		error, success, error_evolution, fes = ant_colony.run(show_info=True)
		error_list.append(error)
		error_evolution_list.append(error_evolution)
		fes_list.append(fes)
		success_count += success

	success_rate = success_count/n_runs
	print(fes_list, error_evolution_list)
	write_performance("%s"%(function_name), error_list, fes_list, fes_list, popSize, success_rate, savePathReporter, max_obj=False)
	write_parameters_acor("%s"%(function_name), popSize, parameters, savePathParameters, max_obj=False)
	write_evolution(error_evolution_list, n_runs, savePathEvolution)
