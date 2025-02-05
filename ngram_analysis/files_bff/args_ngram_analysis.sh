dir_G=$1
dir_Gp=$2
shuffle=$3
expected=$4
filter_size=$5
name_prefix=$6
ngram_size=$7

BFF=../bff/target/release/bff

if [[ -z "$dir_G" || -z "$dir_Gp" || -z "$shuffle" || -z "$expected" || -z "$filter_size" ]] ; then
    echo 'Usage : ./args_ngram_analysis.sh dir_G dir_Gp shuffle expected filter_size [name_prefix] [ngram_size]'
    exit 1
fi

if ! [ -d "$dir_G" ] ; then
    echo Error: "$dir_G" is not a directory
    exit 1
fi

if ! [ -d "$dir_Gp" ] ; then
    echo Error: "$dir_Gp" is not a directory
    exit 1
fi

if ! [[ "$shuffle" = "yes" || "$shuffle" = "no" ]] ; then
    echo Error: shuffle should be \'yes\' or \'no\'
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

if [[ "$ngram_size" =~ $re ]] ; then
    min_=${ngram_size}
    max_=${ngram_size}
else 
    min_=7
    max_=7
fi

files_G=($dir_G/*)
files_Gp=($dir_Gp/*)

# Shuffle ?
if [[ shuffle = yes ]] ; then
    files_G=($(shuf -e "${files_G[@]}"))
    files_Gp=($(shuf -e "${files_Gp[@]}"))
fi

size_G="${#files_G[@]}"
size_Gp="${#files_Gp[@]}"

# # Tronque les données 
# size_G=100
# size_Gp=40

files_G=(${files_G[@]:0:$size_G})  # membres
files_Gp=(${files_Gp[@]:0:$size_Gp})  # non membres

files_Gm=(${files_G[@]:$size_Gp})  # pseudo membres
files_Gnm=(${files_G[@]:0:$size_Gp})  # pseudo non membres

# Gm et Gnm partitionnent G
# G et Gm sont de même taille

echo "G: ${#files_G[@]} ; Gp: ${#files_Gp[@]} ; Gm: ${#files_Gm[@]} ; Gnm: ${#files_Gnm[@]}"

unset G


# Initialize the filter with Gm (members)

if [[ -f bloom.bff ]] ; then
    rm bloom.bff
    echo Removed previous bloom filter
fi

time $BFF \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --whole-document \
    --output-directory "${name_prefix}Gm" \
    ${files_Gm[@]}
    
# mesure Gp (non membres) vs Gm

time $BFF \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --no-update-bloom-filter \
    --whole-document \
    --output-directory "${name_prefix}Gp" \
    ${files_Gp[@]}

# mv overlaps.txt overlaps_Gp.txt

# mesure Gnm (non membres) vs Gm


time $BFF \
    --bloom-filter-file bloom.bff \
    --bloom-filter-size $filter_size \
    --expected-ngram-count $expected \
    --min-ngram-size $min_ \
    --max-ngram-size $max_ \
    --no-update-bloom-filter \
    --whole-document \
    --output-directory "${name_prefix}Gnm" \
    ${files_Gnm[@]}


rm bloom.bff # free memory
