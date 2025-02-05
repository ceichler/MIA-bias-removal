# Picking a subset of the non-members that minimizes the n-gram bias


Pick a subset of the non-members in order to minimise the Kolmogorov-Smirnov distance between their ngram overlap distribution and an optimal unbiased one.
The unbiased ngram overlap distribution is computed from a random sample of members.

#### optim.py
    Test different methods of picking a subset
    (greedy, brute force, genetic algorithm)

#### pick_books.py
    Applies the greedy algorithm on the scores from the BFF
    Outputs `ordered_selection.txt`

#### ordered_selection.txt
    Lists of books by the order in which they were added to the subset by the greedy algorithm.
    Each line is:
    Book ID, path to book, distance (for the subset equal to this book and the previous ones in the list)
    So the best subset found is the one including all books up to the line with the min distance

#### build_dataset.sh
    Makes a directory containing the first books of `ordered_selection.txt` (links to their files) 
    For now, the index of the last book to include is hard coded
    The produced directory is `../../get_data/data/picked_non_members/`
    
