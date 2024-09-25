from Genetic_Algorithm import *
from Snake_Game import *

# n_x -> no. of input units
# n_h -> no. of units in hidden layer 1
# n_h2 -> no. of units in hidden layer 2
# n_y -> no. of output units

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
sol_per_pop = 50
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

# Defining the population size.
pop_size = (sol_per_pop,num_weights)
#Creating the initial population.
new_population = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)

num_generations = 100
with_plot = True
show_plot_step = 20

highlights = []

num_parents_mating = 12
for generation in range(num_generations):
    gen_idx = generation + 1;
    print('############## GENERATION ' + str(gen_idx) + ' ###############' )
    
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_population, gen_idx)
    print('#######  fittest chromosome in gneneration ' + str(gen_idx) +' is having fitness value:  ', np.max(fitness))

    highlights.append(np.max(int(fitness)))

    # Display plot
    if (with_plot and gen_idx % show_plot_step == 0):
        fig = plt.figure()
        plt.plot(np.arange(1, len(highlights) + 1), highlights)
        plt.show()

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation


