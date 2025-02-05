#! /bin/bash

dest_dir="../../get_data/data/picked_non_members/"
rm $dest_dir/*

files=($(cut -d ',' -f 2 ordered_selection.txt))
dists=($(cut -d ',' -f 3 ordered_selection.txt))

min=${dists[0]}
min=1
i_min=0
i=0
echo Init $min.
for val in ${dists[@]}; do
    if [[ $val < $min ]] ; then
        min=$val
        i_min=$i
    fi
    ((i++))
done

last_index=$i_min
echo Index of the last book included: $last_index

# 1041 0.008641169252068846

for file in ${files[@]:0:${last_index}} ;
do  
    file=$(echo $file | sed -e 's|/home/nchampeil/|../|' )
    # echo $file
    link_name="$dest_dir/$(basename $file).link"
    src_name="$(realpath "$file")"
    ln "$src_name" "$link_name"
done

echo done
