
files_G=(/home/nchampeil/get_data/data/stripped_members/*)
files_Gp=(/home/nchampeil/get_data/data/picked_non_members/*)

# Shuffle ?
files_G=($(shuf -e "${files_G[@]}"))
files_Gp=($(shuf -e "${files_Gp[@]}"))


size_G="${#files_G[@]}"
size_Gp="${#files_Gp[@]}"

# # Tronque les données 
# size_G=100
# size_Gp=40

files_G=(${files_G[@]:0:$size_G})   # membres
files_Gp=(${files_Gp[@]:0:$size_Gp})   # non membres

files_Gm=(${files_G[@]:$size_Gp})   # pseudo membres
files_Gnm=(${files_G[@]:0:$size_Gp})   # pseudo non membres

# Gm et Gnm partitionnent G
# G et Gm sont de même taille

echo "G: ${#files_G[@]} ; Gp: ${#files_Gp[@]} ; Gm: ${#files_Gm[@]} ; Gnm: ${#files_Gnm[@]}"
# echo "G: ${files_G[@]} "
# echo " Gp: ${files_Gp[@]} "
# echo " Gm: ${files_Gm[@]} "
# echo " Gnm: ${files_Gnm[@]}"

unset G

# exit

min_=7
max_=7


expected=10000000 # 10M
filter_size=16777216

# expected=100000000 # 100M
# filter_size=134217728

# expected=100000 # 100k
# filter_size=1048576

expected=$1
filter_size=$2
name_prefix=$3

if [[ -z "$expected" || -z "$filter_size" ]] ; then
    echo Usage : ./ngram_analysis.sh expected filter_size output_file max_files randomize
    exit 1
fi 

re='^[0-9]+$'
if ! [[ "$expected" =~ $re ]] ; then
    echo Error: "$expected" is not a number
    exit 1
fi
if ! [[ "$filter_size" =~ $re ]] ; then
    echo Error: "$filter_size" is not a number
    exit 1
fi


# 16777216

    # --bloom-filter-size 8388608 \


# Initialise le filtre avec Gm (membres)

rm bloom.bff

time /home/nchampeil/bff/target/release/bff \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --whole-document \
    --output-directory "${name_prefix}_picked_Gm" \
    ${files_Gm[@]}
    
# mesure Gp (non membres) vs Gnm

time /home/nchampeil/bff/target/release/bff \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --no-update-bloom-filter \
    --whole-document \
    --output-directory "${name_prefix}_picked_Gp" \
    ${files_Gp[@]}

# mv overlaps.txt overlaps_Gp.txt

# mesure Gnm (non membres) vs Gm


time /home/nchampeil/bff/target/release/bff \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --no-update-bloom-filter \
    --whole-document \
    --output-directory "${name_prefix}_picked_Gnm" \
    ${files_Gnm[@]}

# mv overlaps.txt overlaps_Gnm.txt


  # --output-directory ~/stage_inria/invalidation/results_bff/ \

  # ~/stage_inria/invalidation/members/azerty/test/data-00000-of-00001.arrow

  # --filtering-threshold 1.0 \

  # --whole-document # un ngram peur contenir un linebreak
  # --anotate-only ?? osef le résultat 

