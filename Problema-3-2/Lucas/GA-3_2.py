#Função a otimizar:
# f(x, y) == 0.5 - [ (sen²√(x² + y²)) - 0.5 // (1.0 + 0.001(x² + y²))² ]
# s.a -100 ≤ x ≤ 100
# e -100 ≤ y ≤ 100

# GA: Representação: Vetor de 2 números reais (entre -100 e +100)
# Inicialização: aleatória (dist. uniforme)
# Seleção dos pais: torneio dois a dois
# Crossover: BLX-α
# Mutação: creep (normal com média 0 e desvio-padrão 0.2). Probabilidade: 30%
# Restrições: mutação ou crossover estouraram o limite? refazer.
# Substituição: geracional com elitismo
# População: 100
# Critério de parada: 1000 gerações ou ótimo (0) encontrado

import numpy as np
import matplotlib.pyplot as plt

def generateRandomIndividual():
    return np.random.uniform(-100, +100, 2)

def sortByF(ind):
    return ind[1]

def generatePopulation(size):

    pop = []

    for i in range(size):
        ind = generateRandomIndividual()
        pop.append( [ ind, f( ind[0], ind[1] ) ] )
        # Indivíduos da população têm os genes em [0] e o valor de f em [1]

    pop.sort(key = sortByF, reverse = False) # ordena pelo valor de f em ordem crescente
    return pop

def f(x, y):
    a = ( np.sin( np.sqrt(x**2 + y**2) ) ) ** 2 - 0.5
    b = (1.0 + 0.001 * (x**2 + y**2) ) ** 2

    return 0.5 - (a/b)

def tournamentSelection(pop):

    matingPool = []

    while len(matingPool) < len(pop):

        positions = np.random.randint(0, len(pop), 2)
        ind1, ind2 = pop[positions[0]], pop[positions[1]]
        winner = ind1 if ind1[1] <= ind2[1] else ind2 # compara valores de f. escolhe o de menor aptidão
        matingPool.append(winner)

    return matingPool

def crossover(parent1, parent2, alpha, lowerBound, upperBound):
    # Define o BLX-α crossover (ver página 25 do pdf - capítulo 3)

    while True:
        beta = ( np.random.uniform( -alpha, 1 + alpha ) )
        gene0 = parent1[0][0] + beta * (parent2[0][0] - parent1[0][0])
        gene1 = parent1[0][1] + beta * (parent2[0][1] - parent1[0][1])
        # [0][0] ou [0][0] porque o primeiro 0 é para indexar os genes

        if( lowerBound <= gene0 <= upperBound and lowerBound <= gene1 <= upperBound ):
            return [ [gene0, gene1], 0] # não calculo f agora porque será calculado depois da mutação

def mutation(ind, prob, lowerBound, upperBound):

    while True:
        # adds a random value to the gene with a probability prob
        newGenes = [ i + np.random.normal(0, 0.2) if (np.random.uniform(0, 1) < prob) else i for i in ind[0]]

        if( lowerBound <= newGenes[0] <= upperBound and lowerBound <= newGenes[1] <= upperBound ):
            return [ newGenes, f( newGenes[0], newGenes[1] ) ]
            # já calculo o novo valor de f após a mutação

def reproduction(matingPool, mutationProb, alpha, lowerBound, upperBound, eliteSolution):

    newPop = []

    if eliteSolution:
        newPop.append(eliteSolution)

    while len(newPop) < len(matingPool):

        positions = np.random.randint(0, len(matingPool), 2)
        parent1, parent2 = matingPool[positions[0]], matingPool[positions[1]]
        child = crossover(parent1, parent2, alpha, lowerBound, upperBound)
        child = mutation(child, mutationProb, lowerBound, upperBound)

        newPop.append(child)

    newPop.sort(key = sortByF, reverse = False) # ordena pelo valor de f em ordem crescente
    return newPop

def GA(popSize=100, mutationProb=0.3, lowerBound=-100, upperBound=100, alpha=0.5, genCount=1000, optimum=0, elitism=True):

    pop = generatePopulation(popSize)
    count = 0
    best_f = np.inf

    # Listas armazenando aptidões máxima, mínima e média para cada geração.
    # Inicializadas com os valores das pops. iniciais.
    min_fits = [ pop[0][1] ] # primeiro elemento tem menor aptidão (ordenação)
    max_fits = [ pop[-1][1] ] # último elemento tem maior aptidão (ordenação)
    avg_fits = [ np.average( [ ind[1] for ind in pop ] ) ]

    while count < genCount and best_f > optimum:

        eliteSolution = None

        if(elitism):
            eliteSolution = pop[0] # Menor aptidão

        matingPool = tournamentSelection(pop)
        pop = reproduction(matingPool, mutationProb, alpha, lowerBound, upperBound, eliteSolution)

        min_fits.append( pop[0][1] )
        max_fits.append( pop[-1][1] )
        avg_fits.append( np.average( [ ind[1] for ind in pop ] ) )

        best_f = pop[0][1]
        count += 1

    return(min_fits, max_fits, avg_fits, count, pop[0])

if(__name__ == "__main__"):

    min_fits, max_fits, avg_fits, count, best = GA()

    print( "Encontrado o valor de " + str( best[1] ) + " - ponto " + str( best[0] ) + " em " + str(count) + " gerações.")

    time = [ i for i in range(len(min_fits) ) ]

    plt.xlabel("Geração")
    plt.ylabel("f(x)")
    plt.title("Problema 3.2 - Aptidões")
    plt.plot(time, max_fits, 'r', label='Máxima')
    plt.plot(time, min_fits, 'b', label='Mínima')
    plt.plot(time, avg_fits, 'k', label='Média')
    plt.legend(loc='best')
    plt.savefig("3-2_fits.png", bbox_inches="tight")
    plt.clf()
