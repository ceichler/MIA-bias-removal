#!/bin/bash

# ~2 min


input_dir_1=$1
input_dir_2=$2
output_file=$3
max_files=$4
randomize=$5

if [[ -z "$input_dir_1" || -z "$input_dir_2" || -z "$output_file" || -z "$max_files" || -z "$randomize" ]] ; then
    echo Usage : ./files_to_arff.sh input_dir_1 input_dir_2 output_file max_files randomize
    exit 1
fi 

if ! [ -d "$input_dir_1" ] ; then
    echo Error: "$input_dir_1" is not a directory
    exit 1
fi

if ! [ -d "$input_dir_2" ] ; then
    echo Error: "$input_dir_2" is not a directory
    exit 1
fi

re='^[0-9]+$'
if ! [[ "$max_files" =~ $re ]] ; then
    echo Error: "$max_files" is not a number
    exit 1
fi

if ! [[ "$randomize" = "yes" || "$randomize" = "no" ]] ; then
    echo Error: "$randomize" should be \'yes\' or \'no\'
    exit 1
fi

labels=(G G+)

content="
% Poject Gutenberg books with label to try using weka.\n\n\
@relation books\n\
@attribute text string\n\
@attribute label {${labels[0]}, ${labels[1]}}\n\
@data\n
"

echo -e $content > $output_file

add_records() {
    i=1
    for file in ${files[@]} ;
    do
        if [[ $max_files -gt 0 && $i -gt $max_files ]] ; 
        then break ; fi

        # echo -ne "$((i++)) $file"
        echo -ne "$((i++)) $file \r"
        
        echo -n \" >> $output_file
        # sed -e ':a;N;$!ba;s/\r/\\r/g' -e ':a;N;$!ba;s/\n/\\n/g' -e 's/\"/\\\"/g' $file  >> $output_file
        # sed ':a;N;$!ba;s/\r/\\r/g' $file | sed ':a;N;$!ba;s/\n/\\n/g' | sed 's/\"/\\\"/g' >> $output_file
        sed 's/\\/\\\\/g' $file | sed ':a;N;$!ba;s/\r/\\r/g' | sed ':a;N;$!ba;s/\n/\\n/g' | sed 's/\"/\\\"/g' | sed '$ s/\\*$//' >> $output_file
        # sed ':a;N;$!ba;s/\r\n/\\r\\n/g' $file | sed 's/\"/\\\"/g' >> $output_file # OK

        # sed -e ':a;N;$!ba;s/\n/\\n/g' -e ':a;N;$!ba;s/\r/\\r/g' -e 's/\"/\\\"/g' -e '$ s/\\*$//' $file >> $output_file
        truncate -s -1 $output_file # OK


        # sed -e ':a;N;$!ba;s/\r/\\r/g' -e ':a;N;$!ba;s/\n/\\n/g' -e 's/\"/\\\"/g' -e '$ s/\\*$//' $file >> $output_file

        echo \",$label >> $output_file
        echo >> $output_file #OK
    done
    return 0
}


label=${labels[0]}
files=($input_dir_1/*)
if [ "$randomize" = "yes" ] ; then
    files=( $(shuf -e "${files[@]}") )
fi

# files=("members/scripts_ok/data_members_google/1551.txt" "members/scripts_ok/data_members_google/26705.txt" "members/scripts_ok/data_members_google/28653.txt")
# files=("members/scripts_ok/data_members_google/31488.txt" "members/scripts_ok/data_members_google/29145.txt" "members/scripts_ok/data_members_google/32061.txt")


add_records



label=${labels[1]}
files=($input_dir_2/*)
if [ "$randomize" = "yes" ] ; then
    files=( $(shuf -e "${files[@]}") )
fi

add_records