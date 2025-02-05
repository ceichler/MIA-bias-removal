import csv
import argparse

parser = argparse.ArgumentParser(description="Extracts the book ids from the csv")
parser.add_argument("input",
                    help="csv file")
parser.add_argument("output",
                    help="text file")

args = parser.parse_args()

with open(args.input, "r") as src:
    src_reader = csv.reader(src)
    ids = [row[0] + '\n' for row in src_reader]

with open(args.output, "w") as dest:
    dest.writelines(ids)
