#!/bin/bash

# ~4h30

# ~35 min parallel (members)

input_file=$1
output_dir=$2

if ! [ -f $input_file ] ; then
echo Error: $input_file is not a file
exit 1
fi

if ! [ -d $output_dir ] ; then
echo Error: $output_dir is not a directory
exit 1
fi

mapfile -t files < $input_file
# mapfile -t files < ./filtered_names.txt

echo ${files[0]}

len=${#files[@]}
# for (( i=0; i<${len}; i++ )); 
# do
#     echo -ne "${i} / ${len}       \r"
#     wget "https://www.gutenberg.org/cache/epub/${files[${i}]}/pg${files[${i}]}.txt" -P ./data/ -nv
# done

echo parallel

parallel echo -e {1} / ${len} ';' wget https://www.gutenberg.org/cache/epub/{2}/pg{2}.txt -P $output_dir -nv -q ::: $(seq 0 $((${#files[@]} - 1))) :::+ "${files[@]}"  ;
# parallel echo -e {2} - {1}" / ${len}       \r" ';' ::: $(seq 0 $((${#files[@]} - 1))) :::+ "${files[@]}"  ;
