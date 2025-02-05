#!/bin/bash

input_dir=$1
nb_files=$2
output_dir=$3

if [[ -z "$input_dir" || -z "$nb_files" || -z "$output_dir" ]] ; then
    echo Usage : ./sample.sh input_dir nb_files output_dir
    exit 1
fi 

if ! [ -d "$input_dir" ] ; then
    echo Error: "$input_dir" is not a directory
    exit 1
fi

re='^[0-9]+$'
if ! [[ "$nb_files" =~ $re ]] ; then
    echo Error: "$nb_files" is not a number
    exit 1
fi

if ! [ -d "$output_dir" ] ; then
    echo Error: "$output_dir" is not a directory
    exit 1
fi

files=("$input_dir"/*)
files=( $(shuf -e "${files[@]}") )

for file in ${files[@]:0:$nb_files} ;
do 
    link_name="$output_dir/$(basename $file).link"
    src_name="$(realpath "$file")"
    # echo "$src_name"
    # echo "$link_name"
    # if [ -a $"link_name" ] ; then 
    # print existe déjà ??
    # fi
    ln -s "$src_name" "$link_name"
done

# a=(00 11 22 33 44 55 66 77 88 99)

# k=5

# echo xyz
# echo "${a[@]}"
# echo "${a[@]:0:$k}"