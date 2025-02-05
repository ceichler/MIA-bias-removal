# Scripts to collect the initial data from Gutenberg


#### scripts/
    All the scripts used to gather the Gutenberg dataset

#### new_pipeline.sh
    List of script executions that allow the creation of the initial dataset

## Members/Non-Members selection and collection 

Get the list of books in the PG-19 dataset (metadata.csv) and in the current Project Gutenberg catalog


Find the non members (added after the cutoff)  
Keep only the books with original publication date between 1850 and 1910  
Download the books
Remove the boilerplates
Turn it to a .arff dataset

## Scripts/

#### metadata_to_catalog.py
    (for members only)  
    Get the book IDs from PG-19 (`metadata.csv`) and selects the corresponding rows in `pg_catalog.csv` so that the lists for both sets are in the same format  

#### filter_cutoff.py
    (for non-members only)  
    Filters books depending on their date of addition to the library to get a list of non members  

#### parallel_original_publication.py
    Querries OpenLibrary to get an estimation of the publication date  
    Keep only the books released from 1850 to 1910  
    (Can querry GoogleBooks instead, but it gets far worse estimations)
    This takes a long time, but too much parallelisation makes the OpenLibrary servers fail, resulting in errors 503  
    We found that using 4 cpus or less works fine  
    Usage: `mpirun -n N_CPU python3 parallel_original_publication.py INPUT LOGS OUTPUT [-api API]`  
    API is 'ol' for OpenLibrary (default) or 'gb' for GoogleBooks  
    LOGS will contain the date found for each book  

#### filter.py
    Filters books using the date found in the PG-19 metadata.  
    Abbandonned to avoid biases due to a difference of treatment between members and non-members  
    Instead we use `parallel_original_publication.py` as for the non-memebers  

#### csv_to_ids.py
    Extracts the ID numbers from a csv list of books  

#### download_from_gutenberg.sh*
    From their IDs, downloads the ebooks from the Project Gutenberg   

#### download_from_gs.sh*
    (members only)  
    From their IDs, downloads the ebooks from PG-19 (hosted on google cloud storage)  
    Abandonned because would cause time-shift in the book formatting  

#### strip_text.sh*
    Each book has the following structure:  
    Removes the tags and what's outside of them  
    Some text mentionning the PG (not part of the original text) may remain as it was located between the tags  
>    Boilerplate and metadata  
>    \*\*\* START OF THE PROJECT GUTENBERG EBOOK XXX \*\*\*  
>    The book content itself  
>    \*\*\* END OF THE PROJECT GUTENBERG EBOOK XXX\*\*\*  
>    Boilerplate  

#### files_to_arff.sh*
    Usage: ./files_to_arff.sh DIR_1 DIR_2 OUT_FILE N SHUFFLE  
    Creates an .arff dataset of records of the form (text, label)  
    Taking N books from Dir_1 and N books from DIR_2 (or all the books of a dir if N is bigger than that)  
    If N is 0, all books available are used in each dir  
    SHUFFLE is either 'yes' (expected use) or 'no' (for replicability)  


#### mean_size.sh*
    Statistics over the lengthes of the books

#### sample.sh*
    Usage : `./sample.sh input_dir nb_files output_dir`
    Samples files uniformly (without replacement) and creates soft links to the selected files

#### random_partition.sh*
    Takes files from DIR/all/ and splits them randomly between DIR/group_1/ and DIR/group_2/
    Used to prove the blind classifier can learn the biased dataset, not any dataset
    DIR
    ├── all
    ├── group_1
    └── group_2


---

Data collection pipeline:

metadata.csv  
`metadata_to_catalog.py`  
--> pg19_catalog.csv 
 
pg_catalog.csv  
`filter_cutoff.py`  
--> filtered_cutoff.csv  

pg19_catalog.csv or filtered_cutoff.csv  
`parallel_original_publication.py`  
--> (non_)members_filtered_publication.csv  
`csv_to_ids`  
--> (non)members_ids.py  
`download_from_gutenberg.sh`  
--> data_(non_)members_gutenberg/  
`files_to_arff.sh`  
--> data.arff  


