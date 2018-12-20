import random
from engine import TetrisEngine

genes = ['holes_stack_area', 'holes_clean_area', 'height_stack_area',
         'height_clean_area', 'blocked_lines', 'num_stack_area',
         'enemy_blocked_lines']

init_random_value = 10.0

width, height = 10, 20  # standard tetris friends rules


class DNA:
    def __init__(self, mutation_rate, engine):
        self.dict_genes = dict()
        self.dict_genes['holes_stack_area'] = random.uniform(0.0, init_random_value)
        self.dict_genes['holes_clean_area'] = random.uniform(0.0, init_random_value)
        self.dict_genes['height_stack_area'] = random.uniform(0.0, init_random_value)
        self.dict_genes['height_clean_area'] = random.uniform(0.0, init_random_value)
        self.dict_genes['blocked_lines'] = random.uniform(0.0, init_random_value)
        self.dict_genes['num_stack_area'] = random.uniform(0.0, init_random_value)
        self.dict_genes['enemy_blocked_lines'] = random.uniform(0.0, init_random_value)

        self.mutation_rate = mutation_rate
        self.engine = engine
        self.fitness = 0.0
        self.prob = 0.0

    def __str__(self):
        dna_values = self.dict_genes.values()
        dna_str = ', '.join(str(e) for e in dna_values)
        return dna_str

    def calculate_fitness(self):
        self.fitness = sum(self.dict_genes.values())

    def make_sexy_baby(self, parent2):
        baby = DNA(self.mutation_rate, self.engine)
        split_point = int(len(genes)/2)

        # Crossover
        for i in range(len(genes)):
            if i < split_point:
                baby.dict_genes[genes[i]] = self.dict_genes[genes[i]]
            else:
                baby.dict_genes[genes[i]] = parent2.dict_genes[genes[i]]

        # Mutation
        for i in range(len(genes)):
            if random.random() < self.mutation_rate:
                baby.dict_genes[genes[i]] = random.uniform(0.0, init_random_value)

        return baby


class Population:
    def __init__(self, population_size, mutation_rate, engine):
        self.population = list()
        self.current_generation = 0
        self.engine = engine
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best = None
        self.max_fitness = 0.0

        for i in range(population_size):
            self.population.append(DNA(self.mutation_rate, self.engine))

    def calc_fitness_prob(self):
        total = 0
        best = self.population[0]
        for i in range(self.population_size):
            self.population[i].calculate_fitness()
            total += self.population[i].fitness
            if self.population[i].fitness > best.fitness:
                best = self.population[i]
        self.best = best
        self.max_fitness = best.fitness
        for i in range(self.population_size):
            self.population[i].prob = self.population[i].fitness / total

    def _generate_child(self):
        random_DNA1 = self.population[random.randint(0, self.population_size - 1)]
        random_DNA2 = self.population[random.randint(0, self.population_size - 1)]
        random_prob1 = random.random()
        random_prob2 = random.random()
        while random_DNA1.prob > random_prob1 or random_DNA2.prob > random_prob2:
            if random_DNA1.prob > random_prob1:
                random_DNA1 = self.population[random.randint(0, self.population_size - 1)]
                random_prob1 = random.random()
            if random_DNA2.prob > random_prob2:
                random_DNA2 = self.population[random.randint(0, self.population_size - 1)]
                random_prob2 = random.random()
        return random_DNA1.make_sexy_baby(random_DNA2)

    def get_avg_fitness(self):
        avg_fitness = 0.0
        for i in range(self.population_size):
            avg_fitness += self.population[i].fitness
        avg_fitness = avg_fitness / self.population_size
        return avg_fitness

    def generate_next_generation(self):
        next_generation = list()
        for i in range(self.population_size-1):
            next_generation.append(self._generate_child())
        next_generation.append(self.best)
        self.current_generation += 1
        self.population = next_generation


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, engine):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.engine = engine
        self.population = Population(population_size, mutation_rate, engine)

    def evolve_the_beasts(self):
        self.population.calc_fitness_prob()
        for i in range(self.num_generations):
            self.population.generate_next_generation()
            self.population.calc_fitness_prob()
            print("Max fitness of generation ", self.population.current_generation,
                  " is ", self.population.max_fitness)
            # print("Average fitness: ", self.population.get_avg_fitness())
            # print(self.population.best)


if __name__ == '__main__':
    engine = TetrisEngine(width, height)
    darwin = GeneticAlgorithm(100, 0.01, 200, engine)
    darwin.evolve_the_beasts()
