BFF=../../bff/target/release/bff

for i in 1 3
do 

    files_Gm=(../data_m$i/rest_members/*)
    files_Gnm=(../data_m$i/train_members/*)
    files_Gp=(../data_m$i/train_non_members/*)

    size_Gm="${#files_Gm[@]}"
    size_Gnm="${#files_Gnm[@]}"
    size_Gp="${#files_Gp[@]}"


    #-------------------------------



    echo "Gp: ${#files_Gp[@]} ; Gm: ${#files_Gm[@]} ; Gnm: ${#files_Gnm[@]}"

    # # expected=100000 # 100k
    # # filter_size=1048576

    # # expected=10000000 # 10M
    # # filter_size=16777216

    # # expected=100000000 # 100M
    # # filter_size=134217728

    expected=1000000000 # 1G
    filter_size=2147483648


    min_=7
    max_=7

    # Init the filters with rest members

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
        
    # measure train members vs rest members

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


    # measure train non members vs rest members

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
done

echo Done
