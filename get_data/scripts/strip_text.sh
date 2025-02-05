#!/bin/bash

# ~3 min for all 19028 members


# *** START OF THE PROJECT GUTENBERG EBOOK THE WAY OF A MAN ***
# 
# 
# E-text prepared by Suzanne Lybarger, Josephine Paolucci, Joshua
# Hutchinson, and the Project Gutenberg Online Distributed Proofreading Team
# 
# 
# 
# Note: Project Gutenberg also has an HTML version of this
#       file which includes the original illustrations.
#       See 14362-h.htm or 14362-h.zip:
#       (https://www.gutenberg.org/dirs/1/4/3/6/14362/14362-h/14362-h.htm)
#       or
#       (https://www.gutenberg.org/dirs/1/4/3/6/14362/14362-h.zip)


# End of Project Gutenberg's Harper's Young People, December 2, 1879, by Various
# 
# *** END OF THE PROJECT GUTENBERG EBOOK HARPER'S YOUNG PEOPLE, DECEMBER 2, 1879 ***


input_dir=$1
output_dir=$2

if [[ -z $input_dir || -z $output_dir ]] ; then
echo Usage : ./strip_test.sh input_dir output_dir
exit 1
fi 

if ! [ -d $input_dir ] ; then
echo Error: $input_dir is not a directory
exit 1
fi

if ! [ -d $output_dir ] ; then
echo Error: $output_dir is not a directory
exit 1
fi


pattern1='\*\*\* START OF THE PROJECT GUTENBERG EBOOK '
pattern2='\*\*\* END OF THE PROJECT GUTENBERG EBOOK '

files=("$input_dir"/*)
# files=../members/scripts_ok/data/*

failed=()

# files=(${files[@]:0:10})

mention=()

i=0
for file in ${files[@]} ;
do
    echo -ne "$((++i)) / ${#files[@]} \r"
    # echo $file

    file_ok=true    

    # Check if pattern2 exists in the file
    if ! grep -q "$pattern1" $file; then
        # echo "Error: Pattern '$pattern1' not found in $file"
        failed+=("$file")
        file_ok=false
    fi

    # Check if pattern2 exists in the file
    if ! grep -q "$pattern2" $file; then
        # echo "Error: Pattern '$pattern2' not found in $file"
        file_ok=false
    fi

    if [ $file_ok == 'true' ] ;
    then

        output_file=$output_dir/$(basename $file)

        sed "0,/${pattern1}/d" $file | sed "/${pattern2}/,\$d" > $output_file # OK

        if grep -q "Project Gutenberg" $output_file ;
        then
            mention+=("$output_file")
            # echo "Still mentions PG : $(basename $file)"
        fi
    else 
        echo f $file
        failed+=("$file")
    fi

done

echo

n=10
if [[ ${#failed[@]} -eq 0 ]] ; 
then 
    echo "Every file is well formated"
else
    echo "Problem with ${#failed[@]} files:" 
    for f in ${failed[@]:0:$n} ; 
    do
        echo "$f" 
    done

    if [[ ${#failed[@]} -gt $n ]] ;
    then
        echo "And $((${#failed[@]} - $n)) more..."
    fi 
fi

echo

if [[ ${#mention[@]} -eq 0 ]] ; 
then 
    echo "No file contains a mention of the Project Gutenberg"
else
    echo "${#mention[@]} files still contain a mention of the Project Gutenberg:" 
    for f in ${mention[@]:0:$n} ; 
    do
        echo "$f" 
    done

    if [[ ${#mention[@]} -gt $n ]] ;
    then
        echo "And $((${#mention[@]} - $n)) more..."
    fi 
fi


# file=book.txt

# echo '0,/'"$pattern1"'/d'
# echo "0,/***/d"
# sed 0,/a/d ../members/scripts_ok/data/pg10006.txt
#  | sed "/$pattern2/,\$d"


# sed 1,/\$pattern1/d $file | sed '/'"$pattern2"'/,$d'
# sed "0,/${pattern1}/d" $file | sed "/${pattern2}/,\$d" # OK



