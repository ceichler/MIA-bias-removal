import csv
import datetime
import re
from argparse import ArgumentParser

parser = ArgumentParser(description="Filters the input csv to keep only the books issued to PG between the given dates.\nDate format is YYYY[-MM[-DD]]")
parser.add_argument("input_file",
                    help="The csv file to filter")
parser.add_argument("output_file",
                    help="The resulting file")
parser.add_argument("-min", dest="min_date",
                    action="store",
                    help="The XXX date before which the records will be discarded")
parser.add_argument("-max", dest="max_date",
                    action="store",
                    # type=datetime.date,
                    # default=datetime.date.max,
                    help="The YYYY-MM-DD date after which the records will be discarded")

# If neither -min nor -max is specified, this does nothing
args = parser.parse_args()


def process_date(date):
    date_pattern = r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$"

    match = re.match(date_pattern, date)
    if match is None:
        raise ValueError("Error: date format is YYYY[-MM[-DD]]")
    return datetime.datetime(*map(int, match.groups()))


min_date = process_date(args.min_date) if args.min_date is not None else datetime.datetime.min
max_date = process_date(args.max_date) if args.max_date is not None else datetime.datetime.max

assert min_date <= max_date

# if args.min_date is None:

print(min_date, max_date)

# print("A", datetime.date("1900-12-4"))
# exit()

# cutoff = datetime.datetime(year=2019, month=2, day=10)
# article_date = datetime.datetime(year=2023, month=10, day=23)

latest_date = datetime.datetime.min

# Keep the books added after the article was published# False-> 13070 books, True -> 14839 books
# allow_after_article = True

filtered_ids = []

with (open(args.input_file, 'r') as src,
      open(args.output_file, 'w') as dest):

    src_reader = csv.reader(src, delimiter=',')
    dest_writer = csv.writer(dest, delimiter=',')

    line_count = 0
    head = next(src_reader)
    print("Column names are :", ", ".join(head))

    for row in src_reader:
        line_count += 1

        id = int(row[0])
        issue_date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        title = row[3]
        authors = row[5]

        latest_date = max(latest_date, issue_date)

        if min_date <= issue_date <= max_date:
            filtered_ids.append((id, title))
            dest_writer.writerow(row)


print(f"filtered / total = {len(filtered_ids)} / {line_count} = {len(filtered_ids)/line_count:.1%}")

# print(*filtered_ids[:5], sep = '\n')

print(latest_date)
