import numpy as np
from scipy.stats import ks_2samp
import argparse


def read_labeled_data(path):
    with open(path, 'r') as file:
        content = file.read()
    assert content[0] == '('

    lines = content[:-1].split('\n')
    overlaps = (float(line[1:].split(',')[0]) for line in lines)
    overlaps = np.fromiter(overlaps, dtype=float)
    # labels = [line.split('"')[1] for line in content[:-1].split('\n')]
    # labels = np.array(labels)

    return overlaps #, labels



def ks_distance(data1, data2):
    """Calculate the KS distance between two datasets."""
    r = ks_2samp(data1, data2)
    # print(type(r), r)
    return r.statistic


for i in (1, 2):
    Gnm = read_labeled_data(f"./results_data_m{i}/overlaps_e1000000000_s2147483648_n7-7_name_data_m{i}Gnm")
    Gp = read_labeled_data(f"./results_data_m{i}/overlaps_e1000000000_s2147483648_n7-7_name_data_m{i}Gp")
    
    print(f"KS distance for m{i}:")
    print(ks_distance(Gnm, Gp))
