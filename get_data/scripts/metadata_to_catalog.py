import csv
import os

import argparse

parser = argparse.ArgumentParser(description="Identifies the books from PG-19, and gathers their rows from `pg_catalog.csv`.")
parser.add_argument("dir",
                    metavar="DIR",
                    help="Directory containing `metadata.csv` (from PG-19), `pg_catalog.csv` (from Project Gutenberg) and where the output `pg19_catalog.csv` will be written")

args = parser.parse_args()


metadata_path = os.path.join(args.dir, "metadata.csv")
catalog_path = os.path.join(args.dir, "pg_catalog.csv")
output_path = os.path.join(args.dir, "pg19_catalog.csv")

ids = []
with open(metadata_path, 'r') as metadata:
    reader = csv.reader(metadata, delimiter=',')
    ids = [int(row[0]) for row in reader]

ids.sort()
print("Size of PG-19:", len(ids))  # 28752
print(f"min {ids[0]}, max {ids[-1]}")
# print(*ids[:200], sep="\n")


catalog_ids = []
with open(catalog_path, 'r') as catalog:
    reader = csv.reader(catalog, delimiter=',')
    next(reader)  # skip the column titles
    catalog_ids = [int(row[0]) for row in reader]

assert all(catalog_ids[i] <= catalog_ids[i + 1] for i in range(len(catalog_ids) - 1))
print("Catalog is well sorted")

# print( [id  for id in ids if id not in catalog_ids] )  # 4 elts : [28520, 30360, 57479, 57486]
#         # ids.append(id)

catalog_ids.sort()
print("Size of catalog:", len(catalog_ids))  # 73623
print(f"min {catalog_ids[0]}, max {catalog_ids[-1]}")

with (open(catalog_path, 'r') as catalog,
      open(output_path, 'w') as output):

    reader = csv.reader(catalog, delimiter=',')
    next(reader)  # skip the column titles
    writer = csv.writer(output)

    i = 0
    j = 0
    row = [0]  #
    for id in ids:
        i += 1
        # print(f"i = {i}, j={j}")
        # row = next(reader)
        # j += 1
        while int(row[0]) < id:
            row = next(reader)
            j += 1

        if int(row[0]) == id:
            writer.writerow(row)
        else:
            print(f"ID {id} was found in pg-19 but not in the catalog")


# catalog ids are not continuous from 10 to


# with (open("pg_catalog.csv", 'r') as catalog,
#       open("filtered_cutoff.csv", 'w') as dest):

#     src_reader = csv.reader(src, delimiter=',')
#     dest_writer = csv.writer(dest, delimiter=',')
# open metadata

# open catalog


# sort

# out =  [row for row in catalog if n° in metadata]
