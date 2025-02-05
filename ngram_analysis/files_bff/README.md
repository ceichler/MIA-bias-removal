
# N-gram analyses

## First Step: compile bff
Firstly, you need to compile bff and copy the binary here

## Scripts
#### args_ngram_analysis.sh  
    script to load the partition the member set G into G1 et G2 
    with |G2| = |Gp| 
    Loads the filter with the n-grams of G1, (Gm)
    Computes the scores of the books of G2 (Gnm) 
    Computes the scores of the books of Gp (Gp)

#### plot_histo.py  
    Usage: `python3 plot_histo.py DIRECTORY`  
    plots the histograms for of the overlap scores of Gp and of G1  
    Handles the old (scores), and new (scores, path/to/file) formats


#### picked_ngram_analysis.sh  
    Launch the different analyses with parameter variations 


# Results

#### var_filter_size/  
    Varying the size of the bloom filter to show 100M expected n-grams is sufficient  
    The shuffling of G before partitionning between G1 and G2 was disabled  

#### 100M_shuf/  
    5 runs, reshuffled each time to show the results is similar for different random draws  
    expected = 100M  

#### labeled/  
    Used the new version of bff: the file name is also reported, not only the score  
    The selection of the subset of the non-members to remove 7-gram biases is performed based on these scores  


#### picked/  
    With the members and the subset of the non members selected to minimize the KS distance on the 7-grams scores between both distributions  

#### picked_var_ngram/  
    Varying the observed n-gram length (3, 5, 7 and 11 grams)  
    To show that not only the 7-gram bias was changed by the subset picking  

#### 10M_picked_var_ngram/  
    Idem   
    Used expected = 10M instead 


---

The results are files whith names like:  

overlaps_e100000000_s134217728_n7-7_name_picked_Gnm  
overlaps_e{EXPECTED}_s{SIZE}_n{MIN}-{MAX}_name_0_picked_{Gm|Gnm|Gp}  

EXPECTED: expected number of n-grams specified for the filter creation  
SIZE: size of the bloom filter  
MAX: length of the n-grams considered  
MIN: always equal to MAX in our measures  

At the end Gnm / Gp represent respectively the non-members in the simulated ideal dataset / real dataset (poor naming convention)  
Gm is the left out part of the members, these scores doesn't hold meaning our case (since the filter is being updated at the same time)  

