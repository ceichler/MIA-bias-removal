#!/bin/bash

echo "Remove previous data"
rm data_m1/*/pg*.txt

members=(../get_data/data/stripped_members/*)
non_members=(../get_data/data/stripped_non_members/*)

members=($(shuf -e "${members[@]}"))
non_members=($(shuf -e "${non_members[@]}"))

n=1000
echo "Training instances: $n x2"

train_members=(${members[@]:0:$n})
train_non_members=(${non_members[@]:0:$n})

rest_members=(${members[@]:$n})
rest_non_members=(${non_members[@]:$n})


if [[ $(( ${#train_members[@]} + ${#rest_members[@]} )) -ne ${#members[@]} ]]
then 
    echo ${#train_members[@]} + ${#rest_members[@]} != ${#members[@]} 
    exit 1 
fi

if [[ $(( ${#train_non_members[@]} + ${#rest_non_members[@]} )) -ne ${#non_members[@]} ]]
then 
    echo ${#train_non_members[@]} + ${#rest_non_members[@]} != ${#non_members[@]} 
    exit 1 
fi


# exit 0

ln -t "data_m1/train_members" ${train_members[@]}
ln -t "data_m1/train_non_members" ${train_non_members[@]}
ln -t "data_m1/rest_non_members" ${rest_non_members[@]}
ln -t "data_m1/rest_members" ${rest_members[@]}


echo "Building train.arff"
../../get_data/scripts/files_to_arff.sh data_m1/train_members data_m1/train_non_members data_m1/train.arff 0 no

echo "Building rest.arff"
../../get_data/scripts/files_to_arff.sh data_m1/rest_members data_m1/rest_non_members data_m1/rest.arff 0 no

