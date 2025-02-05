BFF
===

Modidied version of an existing project.  

Original: https://github.com/allenai/bff/  

Getting started  
---------------

1. Install Rust on your machine.
    1. `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    2. Add `~/.cargo/bin` to your `PATH` environment variable.
2. Run `cargo build --release`. It places the binary at `target/release/bff`.
3. Run `./target/release/bff --help` to see the available options.


---

```
./target/release/bff \  
    --bloom-filter-file FILENAME \  
    --bloom-filter-size SIZE \  
    --expected-ngram-count NB_EXPECTED \  
    --min-ngram-size MIN \  
    --max-ngram-size MAX \  
    [ --no-update-bloom-filter ]  # set this flag only after preloading the filter with members \  
    --whole-document  # n-grams can span over linebreaks \  
    --output-directory results/ \  
    text_1.txt [text_2.txt ...]  
```



This is used to produce a n-gram analysis over the set of non-members against the set of members  

Inputs are .txt files (in the original, it was .json.gz)  

Output is a text file with a line per input file, of the form:  
(overlap_score, "path/to/the/input/file")  

In a previous version, it was only:  
overlap_score  

The overlap score of a non-member document is the proportion its n-grams that are present in the bloom filter, after it was loaded with member n-grams  
This is only meaningful if the bloom filter was preloaded, and is not updating with the current document  


## Parameters

To have an acceptable rate of false positives, the number of expected n-grams should be specified, as well as the size of the bloom filter.  

Launch the program specifying the expected number of n-grams,   
Note the advised size for the filter, interrupt, and relaunch with this size.  

To (experimentally) estimate an acceptable number, try different values over a few orders of magnitude:  
When increasing the expected number of n-grams (and the corresponding size) doesn't change the result much, then the number of false positives is already negligible, and the value is acceptable  

