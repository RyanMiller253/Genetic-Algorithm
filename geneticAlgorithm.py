import sys
import numpy as np
import lib
import fileinput
import random
import operator
import configparser
import copy

# initialize user inputted variables
chromosome_list = []
num_chromosomes = 0
num_generations = 0
user_selection_percent = 0.0
user_selection_method = 0  # 0 - elitist | 1 - tournament
user_crossover_method = 0  # 0 - k-point | 1 - uniform
user_mutation_percent = 0.0
rate_of_mutation_decrease = 0
file_name = ""
data = []

# class for constructing individual chromosomes


class Chromosome():
    def __init__(self):
        # Initializing genes
        self.range1_low, self.range1_high, self.range2_high, self.range2_low, self.buy_short, self.temp_num, self.fitness, self.num_matches = 0, 0, 0, 0, 0, 0, -5000, 0
        self.range1_low = np.random.normal(loc=0, scale=1.15)
        self.range1_high = self.range1_low
        while self.range1_low == self.range1_high:
            self.range1_high = np.random.normal(loc=0, scale=1.15)
        # assuring lower end of range is smaller than larger end of range
        if self.range1_high < self.range1_low:
            self.temp_num = self.range1_high
            self.range1_high = self.range1_low
            self.range1_low = self.temp_num

        self.range2_low = np.random.normal(loc=0, scale=1.15)
        self.range2_high = self.range2_low
        while self.range2_low == self.range2_high:
            self.range2_high = np.random.normal(loc=0, scale=1.15)
        # assuring lower end of range is smaller than larger end of range
        if self.range2_high < self.range2_low:
            self.temp_num = self.range2_high
            self.range2_high = self.range2_low
            self.range2_low = self.temp_num

        self.buy_short = np.random.randint(0, 2)

# validate that input from configuration file is valid


def validate_input(num_chromosomes, num_generations, user_selection_method, user_selection_percent, user_crossover_method, user_mutation_percent, rate_of_mutation_decrease):
    errorFlag = 0
    if not(0 < num_chromosomes <= 1000):
        print('Number of Chromosomes should be in range [1,1000]')
        errorFlag = 1
    if not(0 <= num_generations <= 1000):
        print('Number of Generations should be in range [1,1000]')
        errorFlag = 1
    if not(user_selection_method == 0 or user_selection_method == 1):
        print('Selection Method should be either 0 or 1')
        errorFlag = 1
    if not(user_crossover_method == 0 or user_crossover_method == 1):
        print('Crossover method should be either 0 or 1')
        errorFlag = 1
    if not(0 < user_selection_percent <= 1):
        print('Percent of chromosomes selected should be in range (0,1]')
        errorFlag = 1
    if not(0 < user_mutation_percent <= 1):
        print('Percent of chromosomes mutated should be in range (0,1]')
        errorFlag = 1
    if not(0 < rate_of_mutation_decrease <= 1):
        print('Rate of mutation percentage decrease should be in range [0,1]')
        errorFlag = 1
    if errorFlag == 1:
        exit()

# retrieve data from input file and store in data variable


def gather_data():
    with open(file_name, 'r') as file:
        data = [[float(x) for x in line.split()] for line in file]
    return data

# calculate fitness of chromosome by running specified chromosome through data file


def calculate_fitness(chromosome_list, data):
    for i in range(len(chromosome_list)):
        chromosome_list[i].fitness = -5000  # initialize fitness to -5000
        # variable for number of matches chromosome achieves
        chromosome_list[i].num_matches = 0
    for i in range(len(chromosome_list)):
        for j in range(len(data)):
            # sum up fitness if chromosome matches
            if (data[j][0] >= chromosome_list[i].range1_low and data[j][0] <= chromosome_list[i].range1_high) and (data[j][1] >= chromosome_list[i].range2_low and data[j][1] <= chromosome_list[i].range2_high):
                if chromosome_list[i].num_matches == 0:
                    chromosome_list[i].fitness = 0
                chromosome_list[i].num_matches = chromosome_list[i].num_matches + 1
                if chromosome_list[i].buy_short == 1:
                    chromosome_list[i].fitness += data[j][2]
                elif chromosome_list[i].buy_short == 0:
                    chromosome_list[i].fitness += (data[j][2])*(-1)
    return chromosome_list


# function for elitist selection
def elitist_select(chromosome_list, percentage):
    # retrieve top X amount of chromosomes from sorted list
    top_chromosomes = sorted(chromosome_list, key=operator.attrgetter(
        'fitness'), reverse=True)[:int(len(chromosome_list)*percentage)]
    return top_chromosomes

# function for tournament select


def tourny_select(chromosome_list, percentage):
    newlist = []
    for _ in range(round(len(chromosome_list)*percentage)):
        temp_chrom = Chromosome()  # instantiate temp chromosome
        num1 = random.randint(0, len(chromosome_list)-1)
        num2 = num1
        while num2 == num1:
            num2 = np.random.randint(0, len(chromosome_list)-1)
        # copy victorious chromosome to temp
        if chromosome_list[num1].fitness > chromosome_list[num2].fitness:
            temp_chrom = copy.deepcopy(chromosome_list[num1])
        else:
            temp_chrom = copy.deepcopy(chromosome_list[num2])
        # append temp chromosome to list of selected chromosomes
        newlist.append(temp_chrom)
    return newlist

# function for uniform crossover


def uniform_crossover(selected_chromosome_list, num_chromosomes_needed):
    newlist = []  # temporary list
    for _ in range(num_chromosomes_needed):
        child_chromosome = Chromosome()  # instantiate child chromosome
        # initilize variable for random chromosome in list
        rand_chrom1 = np.random.randint(0, len(selected_chromosome_list)-1)
        rand_chrom2 = rand_chrom1
        while rand_chrom1 == rand_chrom2:
            rand_chrom2 = np.random.randint(0, len(selected_chromosome_list)-1)
        random_int = np.random.randint(0, 2)
        if random_int == 0:
            child_chromosome.range1_low = selected_chromosome_list[rand_chrom1].range1_low
        else:
            child_chromosome.range1_low = selected_chromosome_list[rand_chrom2].range1_low

        random_int = np.random.randint(0, 2)
        if random_int == 0:
            child_chromosome.range1_high = selected_chromosome_list[rand_chrom1].range1_high
        else:
            child_chromosome.range1_high = selected_chromosome_list[rand_chrom2].range1_high
        # assuring lower end of range is smaller than larger end of range
        if child_chromosome.range1_high < child_chromosome.range1_low:
            child_chromosome.temp_num = child_chromosome.range1_high
            child_chromosome.range1_high = child_chromosome.range1_low
            child_chromosome.range1_low = child_chromosome.temp_num

        random_int = np.random.randint(0, 2)
        if random_int == 0:
            child_chromosome.range2_low = selected_chromosome_list[rand_chrom1].range2_low
        else:
            child_chromosome.range2_low = selected_chromosome_list[rand_chrom2].range2_low

        random_int = np.random.randint(0, 2)
        if random_int == 0:
            child_chromosome.range2_high = selected_chromosome_list[rand_chrom1].range2_high
        else:
            child_chromosome.range2_high = selected_chromosome_list[rand_chrom2].range2_high
        # assuring lower end of range is smaller than larger end of range
        if child_chromosome.range2_high < child_chromosome.range2_low:
            child_chromosome.temp_num = child_chromosome.range2_high
            child_chromosome.range2_high = child_chromosome.range2_low
            child_chromosome.range2_low = child_chromosome.temp_num
        # randomly assigning buy/short gene
        random_int = np.random.randint(0, 2)
        if random_int == 0:
            child_chromosome.buy_short = selected_chromosome_list[rand_chrom1].buy_short
        else:
            child_chromosome.buy_short = selected_chromosome_list[rand_chrom2].buy_short

        newlist.append(child_chromosome)
    return newlist

# function for kpoint crossover


def kpoint_crossover(selected_chromosome_list, num_chromosomes_needed):
    newlist = []  # temp list
    for _ in range(num_chromosomes_needed):
        child_chromosome = Chromosome()  # instantiate child chromosome
        # select random number for chromosome inlist
        rand_chrom1 = np.random.randint(
            0, high=len(selected_chromosome_list)-1)
        rand_chrom2 = rand_chrom1
        while rand_chrom1 == rand_chrom2:
            rand_chrom2 = np.random.randint(
                0, high=len(selected_chromosome_list)-1)
        # assign child chromosome values from parents using kpoint = 2
        child_chromosome.range1_low = selected_chromosome_list[rand_chrom1].range1_low
        child_chromosome.range1_high = selected_chromosome_list[rand_chrom1].range1_high
        child_chromosome.range2_low = selected_chromosome_list[rand_chrom2].range2_low
        child_chromosome.range2_high = selected_chromosome_list[rand_chrom2].range2_high
        child_chromosome.buy_short = selected_chromosome_list[rand_chrom2].buy_short
        newlist.append(child_chromosome)  # append child chromosome to list
    return newlist

# function for mutation


def mutation(percent_mutation, chromosome_list):
    # determine if each gene will be mutated based off of percent mutation
    for i in range(len(chromosome_list)):
        if random.random() <= percent_mutation:
            chromosome_list[i].range1_low = np.random.normal(loc=0, scale=1.15)
        if random.random() <= percent_mutation:
            chromosome_list[i].range1_high = np.random.normal(
                loc=0, scale=1.15)
        if random.random() <= percent_mutation:
            chromosome_list[i].range2_low = np.random.normal(loc=0, scale=1.15)
        if random.random() <= percent_mutation:
            chromosome_list[i].range2_high = np.random.normal(
                loc=0, scale=1.15)
        if random.random() <= percent_mutation:
            chromosome_list[i].buy_short = np.random.randint(0, 2)
        # assuring lower end of range is smaller than larger end of range
        if chromosome_list[i].range1_high < chromosome_list[i].range1_low:
            chromosome_list[i].temp_num = chromosome_list[i].range1_high
            chromosome_list[i].range1_high = chromosome_list[i].range1_low
            chromosome_list[i].range1_low = chromosome_list[i].temp_num
        # assuring lower end of range is smaller than larger end of range
        if chromosome_list[i].range2_high < chromosome_list[i].range2_low:
            chromosome_list[i].temp_num = chromosome_list[i].range2_high
            chromosome_list[i].range2_high = chromosome_list[i].range2_low
            chromosome_list[i].range2_low = chromosome_list[i].temp_num
    return chromosome_list

# function to generate first generation and populate list of chromsomes


def generate_original_chromosome_list(num_chromosomes):
    for _ in range(num_chromosomes):
        chromosome_list.append(Chromosome())
    return chromosome_list
# prints low, mean, and high chromsomes from list


def print_low_mean_high(chromosomes, num_generation):
    chromosomes = sorted(
        chromosomes, key=operator.attrgetter('fitness'), reverse=True)
    sum_fitness = 0
    for i in range(len(chromosomes)):
        sum_fitness += chromosomes[i].fitness
    print('Generation #', num_generation)
    print('Max fitness: ', chromosomes[0].fitness, 'Min fitness: ', chromosomes[len(
        chromosomes)-1].fitness, "Average fitness: ", sum_fitness/len(chromosomes), '\n')
# print genes of supplied chromosome


def print_chromsome(chromosome):
    print('Fittest Chromsome:\n', chromosome.range1_low, '|', chromosome.range1_high,
          '|', chromosome.range2_low, '|', chromosome.range2_high, '|', chromosome.buy_short)
    print('Fitness: ', chromosome.fitness)
# main loop


def generation(data, num_chromosomes, num_generations, select_method, select_percent, crossover_method, mutation_rate, delta_mutation_rate):
    chromosome_list = generate_original_chromosome_list(
        num_chromosomes)  # generate first generation
    #calculate_fitness(chromosome_list, data)
    crossover, selected = [], []  # initalize lists for crossover and selection methods
    if num_chromosomes > 1:
        for i in range(num_generations):
            chromosome_list = calculate_fitness(
                chromosome_list, data)  # calculate fitness
            if (i % 10) == 0:  # print low, mean, and high every tenth generation
                print_low_mean_high(chromosome_list, i)
            # call elitist selction function
            if select_method == 0:
                selected = elitist_select(chromosome_list, select_percent)
                # perform user specified crossover method with elitist selection
                if crossover_method == 0:
                    crossover = kpoint_crossover(
                        selected, len(chromosome_list)-len(selected))
                elif crossover_method == 1:
                    crossover = uniform_crossover(
                        selected, len(chromosome_list)-len(selected))
            # call tournament selection function
            if select_method == 1:
                selected = tourny_select(chromosome_list, select_percent)
                # perform user specified crossover method with tournament selection
                if crossover_method == 0:
                    crossover = kpoint_crossover(
                        selected, len(chromosome_list)-len(selected))
                elif crossover_method == 1:
                    crossover = uniform_crossover(
                        selected, len(chromosome_list)-len(selected))
            # combine lists of selected chromsomes and crossed over chromosomes
            chromosome_list = selected + crossover
            # call mutation on newly combined list
            chromosome_list = mutation(mutation_rate, chromosome_list)
            if not (mutation_rate <= 0.01):
                # decrease mutation rate by specified ratio
                mutation_rate = mutation_rate*rate_of_mutation_decrease
    # calculate fitnees for final generation
    chromosome_list = calculate_fitness(chromosome_list, data)
    # print low mean and high chromsomes after all generations
    print_low_mean_high(chromosome_list, num_generations)
    print_chromsome(sorted(chromosome_list, key=operator.attrgetter(
        'fitness'), reverse=True)[0])  # print highest chromsomes genes


# instantiate configuration file parser
config = configparser.ConfigParser()
config.read('config1.ini')  # select config file to use

# inout data from config file to predefined variables
num_chromosomes = int(config['DEFAULT']['numChromosomes'])
num_generations = int(config['DEFAULT']['numGenerations'])
user_selection_percent = float(config['DEFAULT']['userSelectionPercent'])
# 0 - elitist | 1 - tournament
user_selection_method = int(config['DEFAULT']['userSelectionMethod'])
# 0 - k-point | 1 - uniform
user_crossover_method = int(config['DEFAULT']['userCrossoverMethod'])
user_mutation_percent = float(config['DEFAULT']['userMutationPercent'])
rate_of_mutation_decrease = float(config['DEFAULT']['rateOfMutationDecrease'])
file_name = config['DEFAULT']['fileName']

# validate input from config file
validate_input(num_chromosomes, num_generations, user_selection_method, user_selection_percent,
               user_crossover_method, user_mutation_percent, rate_of_mutation_decrease)
data = gather_data()  # retrieve data from data file
# run main loop for user inputted number of generations
generation(data, num_chromosomes, num_generations, user_selection_method, user_selection_percent,
           user_crossover_method, user_mutation_percent, rate_of_mutation_decrease)
