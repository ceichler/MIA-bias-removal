import csv

"""
Opens the `metadata.csv` file from pg19
Filter out records based on the date (original publication date)
Write the id numbers in `filtered_names.txt`
"""

filtered_ids = []

src_path = "/home/nchampeil/get_data/data/metadata.csv"
dest_path = "/home/nchampeil/get_data/data/filtered_members_names.txt"

with open(src_path, 'r') as src:

    src_reader = csv.reader(src, delimiter=',')

    line_count = 0

    for row in src_reader:
        line_count += 1

        id = row[0]
        # title = row[1]
        og_date = int(row[2])
        # link = row[3]

        if 1850 <= og_date <= 1910:
            filtered_ids.append(id)


print(f"filtered / total = {len(filtered_ids)} / {line_count} = {len(filtered_ids)/line_count:.1%}")

with open(dest_path, "w") as dest:
    # dest.writelines(map(lambda s: f"gs://deepmind-gutenberg/train/{s}.txt\n", filtered_ids))
    dest.write("\n".join(filtered_ids))

print("Fini")
