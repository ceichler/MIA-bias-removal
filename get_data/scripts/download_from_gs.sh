#!/bin/bash

# ~15 min

mapfile -t files < ./filtered_names.txt

echo ${files[0]}

prefix="gs://deepmind-gutenberg/train/"
sufix=".txt"

files=("${files[@]/#/$prefix}")
files=("${files[@]/%/$sufix}")


echo ${files[0]}


gsutil -m cp ${files[@]} ./data_members_google/
