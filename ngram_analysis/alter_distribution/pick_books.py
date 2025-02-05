import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import argparse


def ks_distance(data1, data2):
    """Calculate the KS distance between two datasets."""
    r = ks_2samp(data1, data2)
    # print(type(r), r)
    return r.statistic


def read_labeled_data(path):
    with open(path, 'r') as file:
        content = file.read()
    assert content[0] == '('

    lines = content[:-1].split('\n')
    overlaps = (float(line[1:].split(',')[0]) for line in lines)
    overlaps = np.fromiter(overlaps, dtype=float)
    labels = [line.split('"')[1] for line in content[:-1].split('\n')]
    labels = np.array(labels)

    return overlaps, labels


def greedy_optim(A, B):
    """Find a subset B' of B such that B' has a distribution close to A's distribution."""

    A = A.copy()
    B = B.copy()

    B_prime = np.array([], dtype=float)
    selected_indices = []
    available_indices = list(range(B.size))

    distances = np.zeros_like(B)
    counter = 0
    while len(available_indices) > 0:
        print(counter, end="\r", flush=True)
        counter += 1

        temp = np.zeros(B_prime.size + 1, dtype=float) * np.nan
        temp[:-1] = B_prime
        B_prime = temp

        dists = np.ones_like(B) * float('inf')
        for i in available_indices:
            b = B[i]
            B_prime[-1] = b
            dists[i] = ks_distance(A, B_prime)

        i = np.argmin(dists)
        distances[B_prime.size-1] = dists[i]

        b = B[i]
        B_prime[-1] = b
        # B = np.delete(B, i)
        available_indices.remove(i)
        selected_indices.append(i)

    return np.array(selected_indices), distances


parser = argparse.ArgumentParser()
parser.add_argument("path_A")
parser.add_argument("path_B")
parser.add_argument("outputs_tag")
args = parser.parse_args()



A, labels_A = read_labeled_data(args.path_A)
B, labels_B = read_labeled_data(args.path_B)

# A = A[:500]
# B = B[:100]

sorted_indices_A = np.argsort(A)
A = A[sorted_indices_A]
labels_A = labels_A[sorted_indices_A]

sorted_indices_B = np.argsort(B)
B = B[sorted_indices_B]
labels_B = labels_B[sorted_indices_B]
# the i-th overlap value still matches the i-th label

m = min(A.min(), B.min())
M = max(A.max(), B.max())

# for e in (A, B):
#     plt.hist(e, range=(m, M), bins=25, density=False)


def plot_cdf(Z, **kwargs):
    Z.sort()
    # F = np.cumsum(Z)
    n = np.size(Z)
    plt.plot(Z, np.arange(n)/n, **kwargs)


indices, distances = greedy_optim(A, B)
B_prime = B[indices]
labels = labels_B[indices]

j = np.argmin(distances)
print(j, distances[j])

with open("ordered_selection.txt", 'w') as file:
    # file.writelines(list(map(repr, zip(map(int, indices), map(str, labels), map(float, distances)))))

    # for i in range(np.size(B)):
    #     line = f"{indices[i]}, {labels[i]}, {distances[i]}\n"
    file.write(f"{j}\n")
    file.writelines([f"{indices[i]}, {labels[i]}, {distances[i]}\n" for i in range(np.size(B))])


print("A :", np.size(A))
print("B :", np.size(B))
print("B' :", np.size(B_prime))

# plt.show()

# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig(f'figure{i}.png')

plt.figure()

plot_cdf(A)
plot_cdf(B)
plt.legend(["A", "B"])

# for i in range(j-5, j+5):
for i in range(np.size(B)):
    plot_cdf(B_prime[:i+1], linewidth=1, color="#AAAA")

# plt.title(f"algo: {args.function_name}, dist: {distance}")
plt.savefig(f'figure_pick_books_{args.outputs_tag}.png')

plt.figure()
plt.plot(distances)
plt.savefig(f'figure_pick_books_distances_{args.outputs_tag}.png')
