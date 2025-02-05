import requests
# import datetime
import time
import internetarchive as ia
import csv
# import random

from mpi4py import MPI
import argparse


# procs   4     8
# ~      1h   35 min   open library
# ~                    google books

parser = argparse.ArgumentParser(description="For each book of the input csv, querries a library API to find its year of original release.\nThe resukt of the querry is written to `logs` and the ")
parser.add_argument("input",
                    help="csv file")
parser.add_argument("logs",
                    help="file")
parser.add_argument("output",
                    help="csv file")
parser.add_argument("-api",
                    default="ol",
                    choices=["ol", "gb", "ia"],
                    help="which API to querry: ol = Open Library(default), gb = Google Books, ia = Internet Archive")

args = parser.parse_args()

default_return = 9999, ""

src_count = 0
dest_count = 0
err_count = 0


def querry_google_books(querry_title, querry_author=None):
    global err_count

    url = f"https://www.googleapis.com/books/v1/volumes?q=+intitle:{querry_title}"
    if querry_author is not None:
        url += f"+inauthor:{querry_author}"

    # url = f"https://www.gutenberg.org/ebooks/{ebook_id}"

    try:
        response = requests.get(url)
    except Exception as e:
        print("\nREQUEST ERROR :", type(e))
        print(querry_title)
        err_count += 1
        return default_return

    if response.status_code != 200:
        if response.status_code == 503:
            print("dfg", response)
        # raise(f"ERROR: status_code = {response.status_code}")
        print("\nEr0r status :", response.status_code, type(response.status_code))
        err_count += 1
        return default_return

    data = response.json()

    # print(type(data))
    # print(data.get("fghjk"))

    # totalItems = data.get("totalItems")
    items = data.get("items")

    if items is None:
        print("GB - No results to the querry")
        return default_return

    # print(totalItems, len(items))
    # print(items[0]["volumeInfo"].keys())

    # record = None
    # min_date = datetime.datetime(year=datetime.MAXYEAR, month=12, day=31)
    min_date = 9999

    dismiss_title = 0
    dismiss_author = 0
    for item in items:
        info = item["volumeInfo"]

        title = info.get("title")
        authors = info.get("authors")
        publishedDate = info.get("publishedDate")
        # language = info.get("language")

        if title.lower() != querry_title.lower():
            dismiss_title += 1
            continue

        if (
            querry_author is not None
            and authors is not None
            and querry_author.lower() not in map(lambda s: s.lower(), authors)
        ):

            dismiss_author += 1
            continue

        # print(title, authors, publishedDate, language, sep=" -- ")

        if publishedDate is not None:
            # if len(publishedDate) == 4:
            #     date = datetime.datetime.strptime(publishedDate, "%Y")
            # elif len(publishedDate) == 7:
            #     date = datetime.datetime.strptime(publishedDate, "%Y-%m")
            # elif len(publishedDate) == 10:
            #     date = datetime.datetime.strptime(publishedDate, "%Y-%m-%d")
            # else:

            if len(publishedDate) >= 4 and publishedDate[:4].isdigit():
                date = int(publishedDate[:4])
            else:
                print("\n Unparsed format :", publishedDate)
                continue
                # raise BaseException(publishedDate)

            if min_date > date:
                min_date = date
                # record = item

    info = f"GB - nb: {len(items)}, date: {min_date}, dismiss_title: {dismiss_title}, dismiss_author: {dismiss_author}"

    return min_date, info


def querry_openlibrary(querry_title, querry_author=None, limit=None):
    global err_count

    url = f"https://openlibrary.org/search.json?title={querry_title}"

    if querry_author is not None:
        url += f"&author={querry_author}"
    if limit is not None:
        url += f"&limit={limit}"

    try:
        response = requests.get(url)
    except Exception as e:
        print("REQUEST ERROR :", type(e))
        err_count += 1
        print(querry_title)
        return default_return

    if response.status_code != 200:
        if response.status_code == 503:
            print("dfg", response.text)
        # raise(f"ERROR: status_code = {response.status_code}")
        print("\nEr0r status :", response.status_code)
        print(querry_title)

        err_count += 1
        return default_return

    data = response.json()

    # numFound = data.get("numFound")
    items = data.get("docs")

    if items is None:
        print("OL - No results to the querry")
        return default_return

    # print(totalItems, len(items))
    # print(items[0]["volumeInfo"].keys())

    # record = None
    min_date = 9999
    dismiss_title = 0
    dismiss_author = 0

    for item in items:
        # info = item["volumeInfo"]

        title = item.get("title")
        authors = item.get("authors")
        date = item.get("first_publish_year")
        # publishedDate = item.get("publish_year")
        # language = item.get("language")

        if title.lower() != querry_title.lower():
            dismiss_title += 1
            continue
        if (
            querry_author is not None
            and authors is not None
            and querry_author.lower() not in map(lambda s: s.lower(), authors)
        ):
            dismiss_author += 1
            continue

        # print(title, authors, publishedDate, language, sep=" -- ")

        if date is not None:
            # date = min(publishedDate) # int ou list[int]

            if min_date > date:
                min_date = date
                # record = item

    info = f"OL - nb: {len(items)}, date: {min_date}, dismiss_title: {dismiss_title}, dismiss_author: {dismiss_author}"
    return min_date, info


def querry_internet_archive(querry_title, querry_author=None):
    search_query = f'title:("{querry_title}")'
    if querry_author is not None:
        search_query += f'AND creator:("{querry_author}")'

    results = ia.search_items(search_query)

    print(len(results))
    for item in results:
        print(type(item))
        item_metadata = ia.get_item(item["identifier"]).metadata
        if "date" in item_metadata:
            return item_metadata["date"], ""
        elif "year" in item_metadata:
            return item_metadata["year"], ""
    return default_return



titles = [
    "A Journey to the Western Islands of Scotland",
    "Danny's Own Story",
    "Little Lord Fauntleroy",
    "Disowned",
]
authors = [
    "Johnson, Samuel, 1709-1784",
    "Marquis, Don, 1878-1937 Kemble, E. W. (Edward Windsor), 1861-1933 [Illustrator]",
    "Burnett, Frances Hodgson, 1849-1924",
    "Endersby, Victor A., 1891-1988",
]


def compare(querry_title, querry_author=None):
    print("-", querry_title)
    a = querry_google_books(querry_title, querry_author)
    b = querry_openlibrary(querry_title, querry_author)

    print("google {a} / openlibrary {b}")


# for i in range(4):
#     querry_internet_archive(titles[i])

# compare("A Journey to the Western Islands of Scotland", "Johnson, Samuel, 1709-1784") # -> 1775 ok
# compare("Danny's Own Story", "Marquis, Don, 1878-1937 Kemble, E. W. (Edward Windsor), 1861-1933 [Illustrator]") # -> 2005 non : 1912
# compare("Little Lord Fauntleroy", "Burnett, Frances Hodgson, 1849-1924") # -> 1886 ok 1885
# compare("Disowned", "Endersby, Victor A., 1891-1988") #

# compare("A Journey to the Western Islands of Scotland")


# 2064,Text,2000-02-01,A Journey to the Western Islands of Scotland,en,"Johnson, Samuel, 1709-1784","Scotland -- Description and travel -- Early works to 1800
# 51925,Text,2016-05-01,Danny's Own Story,en,"Marquis, Don, 1878-1937
# 479,Text,2006-01-16,Little Lord Fauntleroy,en,"Burnett, Frances Hodgson, 1849-1924",Conduct of life -- Juvenile fiction
# 29384,Text,2009-07-12,Disowned,en,"Endersby, Victor A., 1891-1988",Science fiction

total_rows = 0
with open(args.input, "r") as src:
    src_reader = csv.reader(src, delimiter=",")
    for row in src_reader:
        total_rows += 1


comm = MPI.COMM_WORLD.Dup()
nb_processes = comm.size
rank = comm.rank

if rank == 0:
    print(f"Runs with {nb_processes} processes\n")


querry = {"ol": querry_openlibrary, "gb": querry_google_books, "ia": querry_internet_archive}[args.api]

filtered_rows = []
querry_result = []


with open(args.input, 'r') as src:
    src_reader = csv.reader(src, delimiter=",")

    for row in src_reader:
        src_count += 1
        # if src_count % nb_processes != rank: # plus lent
        if (
            not (total_rows * rank) // nb_processes
            <= src_count
            <= (total_rows * (rank + 1)) // nb_processes
        ):
            continue

        if rank == 0:
            print(f"{src_count} / {total_rows//nb_processes}", end="\r")

        id = int(row[0])
        # issue_date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        title = row[3]
        authors = row[5]

        date, info = querry(title)
        # if date == 9999:
        #     date, info = querry(title)  # try a second time

        # date, info = querry_google_books(title)
        # if date == 9999:

        # info = ""
        # date = 1900 if random.random() < .1 else 42
        # print("    date", date)
        if 1850 <= date <= 1910:
            # print(title)
            filtered_rows.append(row)
            dest_count += 1
        # print(src_count, dest_count, info, end = "           \r", flush=True)
        # print(src_count, dest_count, info)
        if date != 9999:
            querry_result.append(f"{id}, {date}")
        else:
            querry_result.append(f"{id}, ?")

print(f"Proc {rank} finished")

dest_count = comm.reduce(dest_count, op=MPI.SUM, root=0)
err_count = comm.reduce(err_count, op=MPI.SUM, root=0)

# Chacun écrit à son tour
if rank == 0:
    mode = "w"
    print(f"Before: {src_count} \nAfter: {dest_count} \nErrors: {err_count}")

else:
    mode = "a"

if rank > 0:
    comm.recv(None, source=rank - 1)

print(f"Proc {rank} now writing", end="\r")
with open(args.output, mode=mode) as dest:
    dest_writer = csv.writer(dest, delimiter=",")
    for row in filtered_rows:
        dest_writer.writerow(row)

with open(args.logs, mode=mode) as dest:
    dest.write("\n".join(querry_result) + "\n")

if rank < nb_processes - 1:
    comm.send(0, dest=rank + 1)
