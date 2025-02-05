import numpy as np
import matplotlib.pyplot as plt
import itertools
# from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
import argparse
import random


def ks_distance(data1, data2):
    """Calculate the KS distance between two datasets."""
    r = ks_2samp(data1, data2)
    # print(type(r), r)
    return r.statistic


def greedy_optim(A, B):
    """Find a subset B' of B such that B' has a distribution close to A's distribution."""
    threshold = np.size(B) * .1
    # threshold = 0

    A = A.copy()
    B = B.copy()
    B_prime = np.array([], dtype=float)
    current_ks_distance = float('inf')

    counter = 0

    # while len(B_prime) < np.size(B):
    while np.size(B) > 0:
        # increment the length of B_prime
        temp = np.zeros(B_prime.size + 1, dtype=float) * np.nan
        temp[:-1] = B_prime
        B_prime = temp

        # n = len(B_prime)
        print(counter, end="\r", flush=True)
        counter += 1

        dists = np.zeros_like(B)
        for i in range(B.size):
            b = B[i]
            B_prime[-1] = b
            # B_prime.append(b)
            dists[i] = ks_distance(A, B_prime)
            # B_prime.pop()

        i = np.argmin(dists)
        new_dist = dists[i]

        if new_dist < current_ks_distance or np.size(B_prime) < threshold:
            current_ks_distance = new_dist
            b = B[i]
            B_prime[-1] = b
            # B_prime.append(b)
            B = np.delete(B, i)
            # B[n], B[i] = B[i], B[n]  # puts b in the "deleted" part of B
        else:
            return B_prime

    return B_prime


def brute_force_optim(A, B):
    best_value = float('+inf')
    best_subset = []

    for r in range(len(B) + 1):
        print(r)
        for subset in itertools.combinations(B, r):
            value = ks_distance(A, subset)
            if value > best_value:
                best_value = value
                best_subset = subset

    return best_subset, best_value

    return B


def genetic_optim(A, B):
    f = lambda x: ks_distance(A, x)
    generations = 500
    population_size = 200
    mutation_rate = 0.05

    def create_individual():
        return [random.choice([0, 1]) for _ in range(len(B))]

    def evaluate(individual):
        subset = [B[i] for i in range(len(B)) if individual[i] == 1]
        return f(subset)

    def crossover(ind1, ind2):
        point = random.randint(1, len(B) - 1)
        return ind1[:point] + ind2[point:], ind2[:point] + ind1[point:]

    def mutate(individual):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    population = [create_individual() for _ in range(population_size)]
    best_individual = max(population, key=evaluate)

    for generation in range(generations):
        print(generation, end='\r', flush=True)
        population = sorted(population, key=evaluate, reverse=False)
        next_population = population[:population_size//2]

        while len(next_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size//2], 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            next_population.extend([mutate(offspring1), mutate(offspring2)])

        best_individual = max(next_population, key=evaluate)
        population = next_population

    best_subset = [B[i] for i in range(len(B)) if best_individual[i] == 1]
    return best_subset


optim_functions = {'greedy': greedy_optim,
                   #    'brute': brute_force_optim,
                   "genetic": genetic_optim}

parser = argparse.ArgumentParser()
parser.add_argument('function_name', choices=optim_functions.keys())
parser.add_argument('-seed', '-s', action='store_true')

args = parser.parse_args()

if args.seed:
    np.random.seed(1234)

n_a = 2000
n_b = 1000

d = +5

A = np.random.normal(size=n_a)
B = np.random.uniform(-d, +d, size=n_b)

A.sort()
B.sort()

# A *= 100
# B = A + 1

m = min(A.min(), B.min())
M = max(A.max(), B.max())

# for e in (A, B):
#     plt.hist(e, range=(m, M), bins=25, density=False)


def plot_cdf(Z):
    Z.sort()
    # F = np.cumsum(Z)
    n = np.size(Z)
    plt.plot(Z, np.arange(n)/n)


# def empirical_cdf(data):
#     """Calculate the empirical CDF of A dataset."""
#     sorted_data = np.sort(data)
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#     return sorted_data, cdf

# Example usage:
# A = np.random.normal(loc=0, scale=1, size=100)
# B = np.random.normal(loc=0, scale=1, size=200)

A = A
B = B

optim_function = optim_functions[args.function_name]

B_prime = optim_function(A, B)

distance = ks_distance(A, B_prime)

plt.figure()

plot_cdf(A)
plot_cdf(B)
plot_cdf(B_prime)

plt.legend(["A", "B", "B'"])
plt.title(f"algo: {args.function_name}, dist: {distance}")

print("A :", np.size(A))
print("B :", np.size(B))
print("B' :", np.size(B_prime))

# plt.show()


# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig(f'figure{i}.png')

plt.savefig(f'figure_{args.function_name}.png')
