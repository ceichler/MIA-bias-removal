import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os
import argparse
import re

# Will break if there is a " in one of the filenames

parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("-cdf", action="store_true")

args = parser.parse_args()


def plot(path, cdf=False):
    # path = f"/home/nach_linux/stage_inria/files_bff/overlaps_{Gx}.txt"

    with open(path, 'r') as file:
        content = file.read()
    
    if content[0] == '(':
        lines = content[:-1].split('\n')
        overlaps = (float(line[1:].split(',')[0]) for line in lines)
        labels = (line.split('"')[1] for line in content[:-1].split('\n'))
        # print(next(labels))
    else :
        overlaps = map(float, content.split())
        labels = None
    overlaps = np.fromiter(overlaps, dtype=float)

    # overlaps = np.log(1.00000000000001-overlaps)
    # plt.figure()
    ax = plt.subplot()
    if cdf :
        n = np.size(overlaps)
        overlaps.sort()
        plt.plot(overlaps, np.linspace(0, 1, n))
        plt.xlim(-0.05, 1.05)
        
    else :
        plt.hist(overlaps, range=(0, 1), bins=200)
        ax.set_yscale('log')

    # kde = gaussian_kde(overlaps)
    # x = np.linspace(0, 1, 1000)
    # plt.plot(x, kde(x))
    # plt.xlim(0, 1)
    # plt.
    print(path.name)
    match = re.match("overlaps_e(\d+[GMK]?)_s(\d+)_n(\d+)-(\d+)_name_(.*)$", path.name)
    expected, size, _, ngram, name = match.groups()
    for i, prefix in zip((9, 6, 3), "GMK") :
        expected = re.sub("0"*i+"$", prefix, expected)
    
    for a, b in zip(("Gm", "Gnm", "Gp"), (": members", ": unbiased non-members", ": real non-members")) :
        name = re.sub(a, b, name)
    
    #title = f"exp. tok.: {expected}, n-gram: {ngram}"    
    #plt.title(title)
    #plt.suptitle(name)
    plt.xlabel("Overlap score")
    plt.ylabel("Number of books")


files_var_size = [
    # "overlaps_e1M_s2097152_n7-7_name_MtoGnoshufGm",
    "overlaps_e1M_s2097152_n7-7_name_MtoGnoshufGp",
    "overlaps_e1M_s2097152_n7-7_name_MtoGnoshufGnm",
    # "overlaps_e10M_s16777216_n7-7_name_MtoGnoshufGm",
    "overlaps_e10M_s16777216_n7-7_name_MtoGnoshufGp",
    "overlaps_e10M_s16777216_n7-7_name_MtoGnoshufGnm",
    # "overlaps_e100M_s134217728_n7-7_name_MtoGnoshufGm",
    "overlaps_e100M_s134217728_n7-7_name_MtoGnoshufGp",
    "overlaps_e100M_s134217728_n7-7_name_MtoGnoshufGnm",
    # "overlaps_e1G_s2147483648_n7-7_name_MtoGnoshufGm",
    "overlaps_e1G_s2147483648_n7-7_name_MtoGnoshufGp",
    "overlaps_e1G_s2147483648_n7-7_name_MtoGnoshufGnm",
    ]

files_100M_shuf = [
    # "overlaps_e100M_s134217728_n7-7_name_shuf_1Gm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_1Gnm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_1Gp",
    # "overlaps_e100M_s134217728_n7-7_name_shuf_2Gm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_2Gnm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_2Gp",
    # "overlaps_e100M_s134217728_n7-7_name_shuf_3Gm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_3Gnm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_3Gp",
    # "overlaps_e100M_s134217728_n7-7_name_shuf_4Gm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_4Gnm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_4Gp",
    # "overlaps_e100M_s134217728_n7-7_name_shuf_5Gm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_5Gnm",
    "overlaps_e100M_s134217728_n7-7_name_shuf_5Gp",
]

files_labeled = [
    "overlaps_e100M_s134217728_n7-7_name_labeledGm",
    "overlaps_e100M_s134217728_n7-7_name_labeledGnm",
    "overlaps_e100M_s134217728_n7-7_name_labeledGp",
]

files_picked = [
    # "overlaps_e100000000_s134217728_n7-7_name_0_picked_Gm",
    "overlaps_e100000000_s134217728_n7-7_name_0_picked_Gnm",
    "overlaps_e100000000_s134217728_n7-7_name_0_picked_Gp",
    # "overlaps_e100000000_s134217728_n7-7_name_1_picked_Gm",
    "overlaps_e100000000_s134217728_n7-7_name_1_picked_Gnm",
    "overlaps_e100000000_s134217728_n7-7_name_1_picked_Gp",
]

# files = files_var_size
# files = files_100M_shuf
# files = files_labeled
files = files_picked

# for i in range(0, len(files), 2):
# # for i in range(0, len(files), 1):
#     plt.figure()
#     plot(files[i], cdf=True)
#     # plot(files[i])
#     plot(files[i+1], cdf=True)
#     plt.savefig(f"{files[i]}.png")
#     # plt.legend([])



with os.scandir(args.directory) as files:
    sorted_files = sorted(files, key=lambda f:f.name)
    for file in sorted_files:
        name = file.name
        if "Gm" in name:
            continue
        try :
            plt.figure()
            plot(file, args.cdf)
        except Exception as e : 
            print("\033[31mError", name, "\033[0m")
            raise e
        # plt.savefig(name)


plt.show()
