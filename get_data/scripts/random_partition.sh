#!/bin/bash

target_dir=$1

if [[ -z "$target_dir" ]] ; then
    echo Usage : ./random_partition.sh 
    exit 1
fi 

if ! [ -d "$target_dir" ] ; then
    echo Error: "$target_dir" is not a directory
    exit 1
fi

all="$target_dir"/all/
g_1="$target_dir"/group_1/
g_2="$target_dir"/group_2/

mkdir -p "$g_1" "$g_2" "$all"

files=("$all"/*)
files=( $(shuf -e "${files[@]}") )

nb_files=${#files[@]}

for file in ${files[@]:0:$(($nb_files / 2))} ;
do
    mv -i "$file" "$g_1"
done

for file in ${files[@]:$(($nb_files / 2 ))} ;
do
    mv -i "$file" "$g_2"
done
