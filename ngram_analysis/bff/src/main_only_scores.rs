use ahash::RandomState;
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::Rng;
use serde_json::Value;
use std::clone::Clone;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::available_parallelism;
use threadpool::ThreadPool;
#[allow(unused_imports)]
use unicode_segmentation::UnicodeSegmentation;

use std::fs::File;
use std::sync::Mutex;
use std::thread;
use std::sync::mpsc;
use std::env;
use std::fs;




#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    bloom_filter_file: PathBuf,

    /// The size of the bloom filter in bytes. If the filter already exists, this parameter is
    /// ignored.
    #[arg(long)]
    bloom_filter_size: usize,

    /// The number of expected ngrams. This is used to calculate the optimal number of hashers.
    /// If the filter already exists, this parameter is ignored.
    #[arg(long)]
    expected_ngram_count: usize,

    /// The smallest ngram size to consider. Paragraphs that have fewer than this number of tokens
    /// are not deduplicated and always kept. These ngrams are never added to the bloom filter.
    /// Note that this value only matters if the paragraph has fewer tokens than the max ngram size.
    #[arg(long, default_value_t = 5)]
    min_ngram_size: usize,

    /// The largest ngram size to consider. Paragraphs are deduplicated based on the number of
    /// ngrams of this size that are already present in the bloom filter.
    #[arg(long, default_value_t = 13)]
    max_ngram_size: usize,

    /// If this fraction of ngrams of the max ngram size are already present in the bloom filter,
    /// the paragraph is considered a duplicate and is discarded.
    /// Set this to 0 to never produce any output. This is useful when you want to prime the filter
    /// with some content that should be considered duplicates, without deduplicating that content
    /// itself.
    #[arg(long, default_value_t = 0.80)]
    filtering_threshold: f64,

    /// Whether or not to update the bloom filter. If this is true, the filter is not updated, but
    /// the input is still deduplicated based on the filter. Default is false.
    #[arg(long, default_value_t = false)]
    no_update_bloom_filter: bool,

    /// If this is true, we keep the input intact, but we add an annotation to each document that
    /// explains which spans from the text would have been deleted.
    #[arg(long, default_value_t = false)]
    annotate_only: bool,

    /// If this is true, we only write out document id and source, and annotate which spans would
    /// have been deleted. This produces an attribute file per the llm-data specification.
    #[arg(long, default_value_t = false)]
    annotate_attribute_only: bool,

    /// If you want ngrams to span across paragraph breaks, set this to true.
    /// This also means that bff will only remove a complete document at a time. When this happens
    /// the resulting document will be empty. This also means that deduplication within a document
    /// no longer works. All in all, it might be best to only use this when you're also using
    /// --annotate-only.
    #[arg(long, default_value_t = false)]
    whole_document: bool,

    /// If you want to always match whole paragraphs instead of ngrams, set this to true.
    /// Paragraphs smaller than min_ngram_size will still be excluded.
    #[arg(long, default_value_t = false)]
    whole_paragraphs: bool,

    /// The number of threads to use for processing.
    /// If this is 0, the number of threads is automatically determined.
    #[arg(long, short = 't', default_value_t = 0)]
    threads: usize,

    /// Input files. These are expected to be gzip compressed newline-delimited JSON files with a
    /// "text" field.
    #[arg(index = 1)]
    inputs: Vec<PathBuf>,

    /// Output directory. The output files will have the same name as the input files, but be placed
    /// in this directory.
    #[arg(long, short = 'o')]
    output_directory: PathBuf,
}

// #[allow(dead_code)]
// fn tokenize(s: &str) -> impl Iterator<Item = &str> {
//     s.split_word_bounds().filter(|w| {
//         for c in w.chars() {
//             if !c.is_whitespace() {
//                 return true;
//             }
//         }
//         false
//     })
// }

fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split("")
}

struct BloomFilter {
    bits: Vec<AtomicU32>,
    hash_builder_seeds: Vec<[u64; 4]>, // RandomState does not store its seeds, so we have to store them ourselves.
    hash_builders: Vec<RandomState>,
}

impl BloomFilter {
    const MAGIC: u32 = 0x81F0_F117;
    const VERSION: u32 = 1;

    fn optimal_number_of_hashers(size_in_bytes: usize, expected_elements: usize) -> usize {
        let expected_elements = expected_elements as f64;
        let size_in_bits = (size_in_bytes * 8) as f64;
        let k = (size_in_bits / expected_elements) * (2.0f64.ln());
        k.ceil() as usize
    }

    fn prob_of_false_positive(
        size_in_bytes: usize,
        expected_elements: usize,
        num_hashers: usize,
    ) -> f64 {
        let k = num_hashers as f64;
        let m = (size_in_bytes * 8) as f64;
        let n = expected_elements as f64;
        (1.0 - (1.0 - (1.0 / m)).powf(k * n)).powf(k)
    }

    fn suggest_size_in_bytes(expected_elements: usize) -> usize {
        let mut size_in_bytes = 1024 * 1024;
        while size_in_bytes < usize::MAX / 2
            && Self::prob_of_false_positive(
                size_in_bytes,
                expected_elements,
                Self::optimal_number_of_hashers(size_in_bytes, expected_elements),
            ) > 0.01
        {
            size_in_bytes *= 2;
        }
        size_in_bytes
    }

    fn my_prob_of_false_positive(&self, expected_elements: usize) -> f64 {
        Self::prob_of_false_positive(
            self.size_in_bytes(),
            expected_elements,
            self.hash_builders.len(),
        )
    }

    fn size_in_bytes(&self) -> usize {
        self.bits.len() * size_of::<AtomicU32>()
    }

    fn new(size_in_bytes: usize, num_hashers: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers);
        let mut hash_builders = Vec::with_capacity(num_hashers);
        for _ in 0..num_hashers {
            let seeds = rng.gen::<[u64; 4]>();
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let mut bits = Vec::new();
        let number_of_u32 = size_in_bytes / size_of::<AtomicU32>();
        bits.reserve_exact(number_of_u32);
        for _ in 0..number_of_u32 {
            bits.push(AtomicU32::new(0));
        }

        Self {
            bits,
            hash_builder_seeds,
            hash_builders,
        }
    }

    fn from_file(path: &PathBuf) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(path)?;
        let mut stream = BufReader::new(&mut file);

        let magic: u32 = stream.read_u32::<LittleEndian>()?;
        if magic != Self::MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        let version: u32 = stream.read_u32::<LittleEndian>()?;
        if version != Self::VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid version",
            ));
        }

        let num_hashers: u32 = stream.read_u32::<LittleEndian>()?;
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers as usize);
        let mut hash_builders = Vec::with_capacity(num_hashers as usize);
        for _ in 0..num_hashers {
            let seeds = [
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
            ];
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let number_of_elements = stream.read_u64::<LittleEndian>()?;
        let mut bits = Vec::new();
        bits.reserve_exact(number_of_elements as usize);
        for _ in 0..number_of_elements {
            bits.push(AtomicU32::new(stream.read_u32::<NativeEndian>()?));
        }

        Ok(Self {
            bits,
            hash_builder_seeds,
            hash_builders,
        })
    }

    fn write_to_file(&self, path: &PathBuf) -> io::Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let mut stream = BufWriter::new(&file);

        stream.write_u32::<LittleEndian>(Self::MAGIC)?;
        stream.write_u32::<LittleEndian>(Self::VERSION)?;
        stream.write_u32::<LittleEndian>(self.hash_builder_seeds.len() as u32)?;
        for hash_builder_seed in &self.hash_builder_seeds {
            for seed in hash_builder_seed {
                stream.write_u64::<LittleEndian>(*seed)?;
            }
        }

        stream.write_u64::<LittleEndian>(self.bits.len() as u64)?;
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                self.bits.as_ptr().cast::<u8>(),
                self.bits.len() * size_of::<AtomicU32>(),
            );
            stream.write_all(bytes)?;
        };

        Ok(())
    }

    fn hashes(&self, s: &VecDeque<&str>) -> Vec<u64> {
        self.hash_builders
            .iter()
            .map(|hash_builder| {
                let mut hasher = hash_builder.build_hasher();
                s.hash(&mut hasher);
                hasher.finish()
            })
            .collect()
    }

    fn insert_hashes(&self, hashes: &Vec<u64>) {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            self.bits[index].fetch_or(1 << bit, Ordering::Relaxed);
        }
    }

    #[allow(dead_code)] // use in unit test
    fn insert(&self, s: &VecDeque<&str>) {
        let hashes = self.hashes(s);
        self.insert_hashes(&hashes);
    }

    fn contains_hashes(&self, hashes: &Vec<u64>) -> bool {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            if self.bits[index].load(Ordering::Relaxed) & (1 << bit) == 0 {
                return false;
            }
        }

        true
    }

    #[allow(dead_code)] // use in unit test
    fn contains(&self, s: &VecDeque<&str>) -> bool {
        let hashes = self.hashes(s);
        self.contains_hashes(&hashes)
    }
}

#[allow(clippy::too_many_arguments)] // TODO : abstract parameters into a struct
fn process_file(
    input_file: &PathBuf,
    // output_file: &PathBuf,
    bloom_filter: &Arc<BloomFilter>,
    max_ngram_size: usize,
    min_ngram_size: usize,
    update_bloom_filter: bool,
    filtering_threshold: f64,
    annotate_only: bool,
    annotate_attribute_only: bool,
    whole_document: bool,
    whole_paragraphs: bool,
) -> Result<Vec<f64>, io::Error> {
// ) -> Result<f64, io::Error> {
    // println!("Inside process_file");
    // let input_file = OpenOptions::new()
    //     .read(true)
    //     .write(false)
    //     .create(false)
    //     .open(input_file)?;

    // println!("INPUT : {input_file:?}");
    let text = match fs::read_to_string(input_file) {
        Ok(t) => t ,
        Err(e) => {
            println!("FAIL {input_file:?}"); 
            let v = vec![-1.0];
            return Ok(v);
        }
     };
    // match text {
    //     Ok(_) => (),
    //     Err(_) => {println!("FAIL {input_file:?}"); 
    //     let a = Vec::new();
    //     a.push(-1.0);
    //     Ok(a);
    // }
    // }

    // text=text.unwrap();


        // let reader = BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file));
    // let reader = BufReader::new(input_file);

    // let output_file = OpenOptions::new()
    //     .read(false)
    //     .write(true)
    //     .create(true)
    //     .truncate(true)
    //     .open(output_file)?;
    // println!("After OpenOptions");

    // let mut writer = BufWriter::with_capacity(
    //     1024 * 1024,
    //     GzEncoder::new(output_file, Compression::default()),
    // );
    // println!("After BufWriter");
    
    let mut overlaps = Vec::new();

    // let mut first_line = true;
    // let lines = reader.lines();
    // assert_eq!(lines.len(), 1)
    // for line in lines {
    for _ in 0..1 {
        // println!("Line");
        // assert!(first_line);
        // first_line = false; 
        // // Panic if there is more than one book in the json.

        // let line = line.unwrap();
        
        // let mut data: Value = serde_json::from_str(&line).unwrap();
        // let text = data["text"].as_str().unwrap();
        // let text = line;
        // println!("text {text:?}");

        let newlines = if whole_document {  // split doc en paragraphes
            vec![0, text.len()]
        } else {
            let mut newlines = Vec::new();
            newlines.push(0);
            for i in text.match_indices('\n') {
                newlines.push(i.0);
            }
            newlines.push(text.len());
            newlines
        };
        let mut windows_to_remove = Vec::new();
        let mut total_contained_ngrams = 0;
        let mut total_number_of_ngrams = 0;
        
        // println!("len window {}", newlines.windows(2).len());
        // let mut p = 0;
        for paragraph_window in newlines.windows(2) {
            // p += 1;
            // if p > 10 {continue}
            let paragraph = &text[paragraph_window[0]..paragraph_window[1]];
            // println!("paragraph_window {paragraph_window:?}");
            // println!("paragraph {paragraph:?}");


            // calculate hashes for the paragraph
            let mut hashes: Vec<Vec<u64>> = Vec::new();
            let mut ngram: VecDeque<&str> = VecDeque::with_capacity(max_ngram_size);
            for token in tokenize(paragraph) {
                // println!("   token {token:?}");

                ngram.push_back(token);
                // If not hashing whole paragraphs, add ngrams to the bloom filter as they reach max size
                if !whole_paragraphs && ngram.len() >= max_ngram_size { // ajoute à hashes les n_grams de taille max_ngram_size
                    hashes.push(bloom_filter.hashes(&ngram));
                    ngram.pop_front();
                }
            }
            // If the paragraph was too short, put in a shorter ngram, so we can dedupe short
            // paragraphs exactly.
            if hashes.is_empty() && ngram.len() >= min_ngram_size {
                hashes.push(bloom_filter.hashes(&ngram)); // ajoute à hashes le pararagraphe de taille entre min_size et max_size
            }

            let contained_ngrams = hashes
                .iter()
                .filter(|ngram| bloom_filter.contains_hashes(ngram))
                .count();
            total_contained_ngrams += contained_ngrams;

            // calculate how many ngrams are in the bloom filter
            let number_of_ngrams = hashes.len();
            total_number_of_ngrams += number_of_ngrams;

            // produce output
            let too_many_duplicate_ngrams = false &&
                contained_ngrams as f64 / number_of_ngrams as f64 > filtering_threshold;
            if too_many_duplicate_ngrams {
                windows_to_remove.push(paragraph_window);
            } else if update_bloom_filter {
                for ngram in hashes {
                    bloom_filter.insert_hashes(&ngram);
                }
            }
        }

        // if annotate_attribute_only or annotate_only, add the annotation to the json
        // if annotate_attribute_only || annotate_only {
        //     data["bff_duplicate_spans"] = serde_json::to_value(windows_to_remove).unwrap();
        //     data["bff_contained_ngram_count"] =
        //         serde_json::to_value(total_contained_ngrams).unwrap();
        // } else {
        //     let mut output_paragraphs = String::new();
        //     let mut last_end = 0;
        //     for paragraph_window in windows_to_remove {
        //         output_paragraphs.push_str(&text[last_end..paragraph_window[0]]);
        //         last_end = paragraph_window[1];
        //     }
        //     output_paragraphs.push_str(&text[last_end..]);
        //     data["text"] = Value::String(output_paragraphs);
        //     data["bff_contained_ngram_count_before_dedupe"] =
        //         serde_json::to_value(total_contained_ngrams).unwrap();
        // }

        // if annotate_attribute_only {
        //     // Allowed fields
        //     let allowed_fields = [
        //         "bff_duplicate_spans",
        //         "bff_contained_ngram_count",
        //         "id",
        //         "source",
        //     ];

        //     // Iterate through the keys of the JSON object and remove any field that is not in the allowed_fields list
        //     if let Value::Object(ref mut map) = data {
        //         map.retain(|key, _| allowed_fields.contains(&key.as_str()));
        //         }
            
        // }

        // serde_json::to_writer(&mut writer, &data)?;
        // writer.write_all(b"\n")?;

        let overlap = total_contained_ngrams as f64 / total_number_of_ngrams as f64;
        // println!("Score: {}", overlap);
        overlaps.push(overlap)
    }




    // return overlaps;
    // Ok(())
    // Ok(overlaps[0])
    Ok(overlaps)
}

fn main() {
    let args = Args::parse();
    let threads = if args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        args.threads
    };

    let bloom_filter = if args.bloom_filter_file.exists() {
        println!("Loading bloom filter from {:?}...", args.bloom_filter_file);
        BloomFilter::from_file(&args.bloom_filter_file).unwrap()
    } else {
        println!("Creating new bloom filter...");
        let num_hashers = BloomFilter::optimal_number_of_hashers(
            args.bloom_filter_size,
            args.expected_ngram_count,
        );
        BloomFilter::new(args.bloom_filter_size, num_hashers)
    };
    let bloom_filter = Arc::new(bloom_filter);
    println!(
        "Bloom filter loaded. ({} hashers)",
        bloom_filter.hash_builders.len()
    );

    let p = bloom_filter.my_prob_of_false_positive(args.expected_ngram_count);
    if p >= 0.5 {
        println!(
            "WARNING: Probability of a false positive after {} elements is {}.",
            args.expected_ngram_count, p
        );
    } else {
        println!(
            "Probability of a false positive after {} elements: {}",
            args.expected_ngram_count, p
        );
    }

    let suggested_size = BloomFilter::suggest_size_in_bytes(args.expected_ngram_count);
    if suggested_size * 2 < bloom_filter.size_in_bytes() {
        println!(
            "WARNING: Your bloom filter is more than twice as large as suggested for {} elements. \
            This is good for accuracy, but it is much slower, and likely not worth the trade-off.",
            args.expected_ngram_count
        );
    }
    println!(
        "Suggested Size: {}, current/suggested: {}",
        suggested_size, (bloom_filter.size_in_bytes() as f64)/(suggested_size as f64)
    );

    println!("OUTPUT dir : {:?}", args.output_directory.display());

    let len_inputs = args.inputs.len();
    println!("inputs : {}", len_inputs);
    println!("whole doc : {}", args.whole_document);
    println!("no update : {}", args.no_update_bloom_filter);

    // return ();
    
    // let mut results_list = Vec::new();
    // let mut results_list : Vec<Vec<f64>> = Vec::with_capacity(len_inputs);
    let mut results : Vec<f64> = Vec::with_capacity(len_inputs);
    // const size = len_inputs;
    // let results: [f64; size] = [0; size];  // Size should be known at compile time

    // for _ in 0..5 {
    //     // println!("i = {}", i);
    //     results_list.push(Vec::new());
    // }

    
    // let a:ThreadPool::new(args.inputs)

    let threadpool = ThreadPool::new(threads);
    let (tx, rx) = mpsc::channel();
    let rx = Arc::new(Mutex::new(rx));

    let mut i = 0;
    for input in args.inputs {
        // println!("inputs[{}] : {}", i, input.display());
        i += 1;
        // continue;

    // for i in 0..len_inputs {
        // let input = &args.inputs[i];

        let tx = tx.clone();

        let mut output = args.output_directory.clone();
        output.push(input.file_name().unwrap());
        let bloom_filter = bloom_filter.clone();

        // println!("OUTPUT : {output:?}");

        threadpool.execute(move || {
            // if std::thread::current().id() as i32 == 0 {
            if i%10 == 0 {
                print!("{} / {}   \r", i, len_inputs);
                io::stdout().flush();
            }
                // println!("\n Input={input:?} ");
            // };
            let r = process_file(
                &input,
                // &output,
                &bloom_filter,
                args.max_ngram_size,
                args.min_ngram_size,
                !args.no_update_bloom_filter,
                args.filtering_threshold,
                args.annotate_only,
                args.annotate_attribute_only,
                args.whole_document,
                args.whole_paragraphs,
            )
            .unwrap();
            // results_list[i] = results;
            // results[i] = r;
            tx.send(r).expect("TX Failed to send output");
            // println!("Opened {input:?} successfully, result=x");
            // println!("aa");
        });
        // println!("AA");
        // i += 1;
    }
    drop(tx); // Close the sending end of the channel
    println!("Join");
    threadpool.join();

    let outputs: Vec<Vec<f64>> = rx.lock().unwrap().iter().collect();
    // println!("Collected outputs: {:?}", outputs);

    let result_file = format!("/home/nchampeil/files_bff/overlaps_e{}_s{}_n{}-{}_name_{}",
    args.expected_ngram_count, args.bloom_filter_size, args.min_ngram_size, args.max_ngram_size, args.output_directory.display());

    // println!("FILE = {}", result_file);

    // return;
    // let result_file = "/home/nchampeil/files_bff/overlaps.txt";
    // result_file + 1;
    let mut file = OpenOptions::new()
        // .append(true)
        .open(result_file.clone())
        .unwrap_or_else(|_| {println!("CREATE FILE");File::create(result_file).expect("Failed to create file")});
    println!("Overlaps:");
    for list in outputs {
        for nb in list {
            writeln!(file, "{}", nb);
        }
    }

    if !args.no_update_bloom_filter {
        println!("Writing bloom filter to {:?}...", args.bloom_filter_file);
        bloom_filter.write_to_file(&args.bloom_filter_file).unwrap();
        println!("Bloom filter written.");
    }
}
