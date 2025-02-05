#!/bin/bash
n=$(ls  data/ -1 | wc -l)

max_bytes=0
total_bytes=0

for f in data/* ;
do 
    bytes=$(wc -c < $f)
    ((total_bytes += bytes))
    if (( max_bytes < bytes )); then
        max_bytes=$bytes
    fi
done

mean_bytes=$((total_bytes / n))

echo "nb_documents: $n"
echo "total_bytes:  $total_bytes"
echo "mean_bytes:   $mean_bytes"
echo "max_bytes:    $max_bytes"
