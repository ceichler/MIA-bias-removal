

# members : 

wget -P data/ https://storage.googleapis.com/deepmind-gutenberg/metadata.csv
wget -P data/ https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv

# rm -I data/data_members_gutenberg/* ;
python3 scripts/metadata_to_catalog.py data/ &&
mpirun -n 4 --use-hwthread-cpus python3 scripts/parallel_original_publication.py data/pg19_catalog.csv data/logs.txt data/members_filtered_publication.csv -api ol &&
python3 scripts/csv_to_ids.py data/members_filtered_publication.csv data/members_ids.txt &&
scripts/download_from_gutenberg.sh data/members_ids.txt data/data_members_gutenberg/ &&
scripts/strip_text.sh data/data_members_gutenberg/ data/stripped_members/


# non members :
python3 scripts/filter_cutoff.py data/pg_catalog.csv data/filtered_cutoff.csv -min 2019-02-10
mpirun --use-hwthread-cpus python3 scripts/parallel_original_publication.py data/filtered_cutoff.csv data/logs_og_nm.txt data/non_members_filtered_publication.csv -api ol &&
python3 scripts/csv_to_ids.py data/non_members_filtered_publication.csv data/non_members_ids.txt &&
scripts/download_from_gutenberg.sh data/non_members_ids.txt data/data_non_members_gutenberg/ &&
scripts/strip_text.sh data/data_non_members_gutenberg/ data/stripped_non_members/

scripts/files_to_arff.sh data/data_members_gutenberg/ data/data_non_members_gutenberg/ data/data.arff 100 yes
