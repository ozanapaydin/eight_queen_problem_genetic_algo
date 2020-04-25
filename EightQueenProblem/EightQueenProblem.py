import random

import numpy as np

POPULATION_SIZE = 8
TARGET_CHROMOSOME = [0, 6, 3, 5, 7, 1, 4, 2]
MUTATION_RATE = 0.25
TOURNAMENT_SELECTION_SIZE = 4
ORDER_1_CROSSOVER_START_INDEX = 2
ORDER_1_CROSSOVER_FINISH_INDEX = 6
CROSSOVER_CHECK_GENE = -1


class Chromosome:
    def __init__(self):
        self.genes = []
        self.fitness = 0
        i = 0
        while i < TARGET_CHROMOSOME.__len__():
            self.genes = \
                np.random.choice(np.arange(0, POPULATION_SIZE), replace=False, size=[1, POPULATION_SIZE]).tolist()[0]
            # self.genes = np.array(tmp_np_arr).tolist()
            i += 1

    def get_genes(self):
        return self.genes

    def set_genes(self, genes):
        self.genes = genes

    def get_fitness(self):
        self.fitness = 0
        for i in range(self.genes.__len__()):
            if self.genes[i] == TARGET_CHROMOSOME[i]:
                self.fitness += 1
        return self.fitness

    def __str__(self):
        return self.genes.__str__()


class Population:
    # size = size of Population
    def __init__(self, size):
        self.chromozomes = []
        i = 0
        while i < size:
            self.chromozomes.append(Chromosome())
            i += 1

    def get_chromozomes(self):
        return self.chromozomes


def _print_population(population, gen_number):
    print("__________________________")
    print("Generation : ", gen_number, "Fitness Value : ", population.get_chromozomes()[0].get_fitness())
    print("Target Chromozome : ", TARGET_CHROMOSOME)
    i = 0
    for i in population.get_chromozomes():
        print("Chromozome  : ", i, "\tFitness : ", i.get_fitness())


class GeneticAlgorithm:
    @staticmethod
    def evolve(population):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(population))

    @staticmethod
    def _crossover_population(population):
        crossover_population = Population(0)
        i = 0
        crossover_population.get_chromozomes().append(population.get_chromozomes()[i])
        while i < POPULATION_SIZE:
            cr1 = GeneticAlgorithm._select_tournament_population(population).get_chromozomes()[0]
            cr2 = GeneticAlgorithm._select_tournament_population(population).get_chromozomes()[0]
            crossover_population.get_chromozomes().append(GeneticAlgorithm._crossover_chromosome(cr1, cr2))
            i += 1
        return crossover_population

    @staticmethod
    def _mutate_population(population):
        for i in range(POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosome(population.get_chromozomes()[i])
        return population

    @staticmethod
    def _mutate_chromosome(chromosome):
        for i in range(TARGET_CHROMOSOME.__len__()):
            if random.random() < MUTATION_RATE:
                while True:
                    index1 = random.randint(0, TARGET_CHROMOSOME.__len__() - 1)
                    index2 = random.randint(0, TARGET_CHROMOSOME.__len__() - 1)
                    if index1 != index2:
                        chromosome.get_genes()[index1], chromosome.get_genes()[index2] = \
                            chromosome.get_genes()[index2], chromosome.get_genes()[index1]
                        break
        return chromosome

    @staticmethod
    def _crossover_chromosome(cr1, cr2):
        crossover_chromosome_genes = np.full(POPULATION_SIZE, -1).tolist()
        crossover_chromosome_genes[ORDER_1_CROSSOVER_START_INDEX:ORDER_1_CROSSOVER_FINISH_INDEX] \
            = cr1.get_genes()[ORDER_1_CROSSOVER_START_INDEX:ORDER_1_CROSSOVER_FINISH_INDEX]

        i = ORDER_1_CROSSOVER_FINISH_INDEX

        while 1:
            if CROSSOVER_CHECK_GENE not in crossover_chromosome_genes:
                break

            index = i % TARGET_CHROMOSOME.__len__()
            if cr2.get_genes()[index] not in crossover_chromosome_genes:
                val = cr2.get_genes()[index]
                if crossover_chromosome_genes[index] == -1:
                    crossover_chromosome_genes[index] = cr2.get_genes()[index]
                else:
                    while crossover_chromosome_genes[index] != -1:
                        index += 1
                        index = index % TARGET_CHROMOSOME.__len__()
                    crossover_chromosome_genes[index] = val
            i += 1

        crossover_chromosome = Chromosome()
        crossover_chromosome.set_genes(crossover_chromosome_genes)
        return crossover_chromosome

    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromozomes().append(pop.get_chromozomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_chromozomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop


if __name__ == '__main__':
    population = Population(POPULATION_SIZE)  # beginning population
    population.get_chromozomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    _print_population(population, 0)
    generation_number = 2

    while population.get_chromozomes()[0].get_fitness() < TARGET_CHROMOSOME.__len__():
        population = GeneticAlgorithm.evolve(population)
        population.get_chromozomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        _print_population(population, generation_number)
        generation_number += 1
