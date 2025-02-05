import csv
import matplotlib.pyplot as plt
import numpy as np

with open("ordered_selection.txt", "r") as file:
    reader = csv.reader(file)
    dists = [float(row[2]) for row in reader]

print("Last index to include in the subset:", np.argmin(dists))

plt.plot(dists)
plt.xlabel("Nombre de livres ajoutés")
plt.ylabel("Distance à la cible")
plt.savefig("figure_pick_books_distances.png")
plt.show()
