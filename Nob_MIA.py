# Code partially adapted from https://github.com/pratyushmaini/llm_dataset_inference

from metrics import get_aggregate_metrics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from sklearn.metrics import roc_auc_score
import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()}")    
else:
    print("GPU not available, using CPU")

def write(values):
    with open("results_roc_auc.csv", mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        scores_writer.writerow(values)
    scores_file.close() 

def prepare_model(model_name, cache_dir, quant=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 512

    if quant is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True).cuda()
    elif quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    elif quant == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, load_in_8bit=True).cuda()

    print("Model loaded")
    return model, tokenizer

def normalize_and_stack(train_metrics, val_metrics, normalize="train"):
    '''
    excpects an input list of list of metrics
    normalize val with corre
    '''
    new_train_metrics = []
    new_val_metrics = []
    for (tm, vm) in zip(train_metrics, val_metrics):
        if normalize == "combined":
            combined_m = np.concatenate((tm, vm))
            mean_tm = np.mean(combined_m)
            std_tm = np.std(combined_m)
        else:
            mean_tm = np.mean(tm)
            std_tm = np.std(tm)
        
        if normalize == "no":
            normalized_vm = vm
            normalized_tm = tm
        else:
            #normalization should be done with respect to the train set statistics
            normalized_vm = (vm - mean_tm) / std_tm
            normalized_tm = (tm - mean_tm) / std_tm
        
        new_train_metrics.append(normalized_tm)
        new_val_metrics.append(normalized_vm)

    train_metrics = np.stack(new_train_metrics, axis=1)
    val_metrics = np.stack(new_val_metrics, axis=1)
    return train_metrics, val_metrics

def prepare_metrics(members_metrics, nonmembers_metrics):
    keys = list(members_metrics.keys())
    np_members_metrics = []
    np_nonmembers_metrics = []
    for key in keys:
        members_metric_key = np.array(members_metrics[key])
        nonmembers_metric_key = np.array(nonmembers_metrics[key])

        np_members_metrics.append(members_metric_key)
        np_nonmembers_metrics.append(nonmembers_metric_key)

    # concatenate the train and val metrics by stacking them
    np_members_metrics, np_nonmembers_metrics = normalize_and_stack(np_members_metrics, np_nonmembers_metrics)

    return np_members_metrics, np_nonmembers_metrics

# aux functions about MIA classifier

def get_dataset_splits(_train_metrics, _val_metrics, num_samples):
    # get the train and val sets
    for_train_train_metrics = _train_metrics[:num_samples]
    for_train_val_metrics = _val_metrics[:num_samples]
    for_val_train_metrics = _train_metrics[num_samples:]
    for_val_val_metrics = _val_metrics[num_samples:]


    # create the train and val sets
    train_x = np.concatenate((for_train_train_metrics, for_train_val_metrics), axis=0)
    train_y = np.concatenate((-1*np.zeros(for_train_train_metrics.shape[0]), np.ones(for_train_val_metrics.shape[0])))
    val_x = np.concatenate((for_val_train_metrics, for_val_val_metrics), axis=0)
    val_y = np.concatenate((-1*np.zeros(for_val_train_metrics.shape[0]), np.ones(for_val_val_metrics.shape[0])))
    
    # return tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    
    return (train_x, train_y), (val_x, val_y)

def train_model(inputs, y, num_epochs=10000):
    num_features = inputs.shape[1]
    model = get_model(num_features)
        
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert y to float tensor for BCEWithLogitsLoss
    y_float = y.float()

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Squeeze the output to remove singleton dimension
            loss = criterion(outputs, y_float)
            loss.backward()
            optimizer.step()
            pbar.set_description('loss {}'.format(loss.item()))
    return model

def get_model(num_features, linear = True):
    if linear:
        model = nn.Linear(num_features, 1)
    else:
        model = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Single output neuron
        )
    return model

def get_predictions(model, val, y):
    with torch.no_grad():
        preds = model(val).detach().squeeze()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(preds, y.float())
    return preds.numpy(), loss.item()

def read_data(input_file):

    if input_file.split('.')[-1] == 'csv':
        df = pd.read_csv(input_file)
        texts_with_label_member = df[df['label'] == 'G']['text'].tolist()
        texts_with_label_nonmember = df[df['label'] == 'G+']['text'].tolist()

    else:
        # Separate member/non-member datasets
        texts_with_label_member = []
        texts_with_label_nonmember = []

        # Step 1: Read the file and process line by line
        with open(input_file, 'r', encoding='utf-8') as file:
            data_section_started = False
            for line in file:
                # Check if we have reached the @data section
                if line.strip().lower() == '@data':
                    data_section_started = True
                    continue
                
                # Skip lines until we reach the @data section
                if not data_section_started:
                    continue

                # Process each line in the data section
                # Remove quotes and split by commas
                line = line.replace('"', '').strip()
                if line:
                    # Assuming the format is: "text, label"
                    text, label = line.rsplit(',', 1)
                    if label.strip() == 'G':
                        texts_with_label_member.append(text.strip())
                    else:
                        texts_with_label_nonmember.append(text.strip())
    
    return texts_with_label_member, texts_with_label_nonmember

def get_mia_roc(A_members, A_nonmembers, model_name, cache_dir, batch_size, ds_name, metric_list):

    if type(metric_list) == list:
        pass
    else:
        metric_list = [metric_list]

    # quantization options: None, fp16, 8bit (needs accelerate)
    llm, tokenizer = prepare_model(model_name, cache_dir=cache_dir, quant="fp16")

    # MIA
    common_fpr = np.arange(0, 1.009, 0.01)
    lst_tpr = []
    lst_fpr = []
    lst_auc_roc = []
    
    for seed in range(5):
        A_members_metrics = get_aggregate_metrics(llm, tokenizer, A_members, metric_list, None, batch_size=batch_size)
        A_nonmembers_metrics = get_aggregate_metrics(llm, tokenizer, A_nonmembers, metric_list, None, batch_size=batch_size)
        train_metrics, val_metrics  = prepare_metrics(A_members_metrics, A_nonmembers_metrics)

        # How many samples to use for training and validation?
        num_samples = 250 

        np.random.shuffle(train_metrics)
        np.random.shuffle(val_metrics)

        # train a model by creating a train set and a held out set
        (train_x, train_y), (val_x, val_y) = get_dataset_splits(train_metrics, val_metrics, num_samples)

        model = train_model(train_x, train_y, num_epochs = 1000)        
        preds, _ = get_predictions(model, val_x, val_y)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(val_y, preds)

        # Compute AUC
        roc_auc = auc(fpr, tpr)  

        # Interpolate to the common_fpr
        interp_tpr = np.interp(common_fpr, fpr, tpr)
        
        # Append results
        lst_tpr.append(interp_tpr)
        lst_auc_roc.append(roc_auc)
        print('seed:', seed, " ROCAUC:", roc_auc)
        write([model_name, ds_name, seed, metric_list, interp_tpr, roc_auc])
    
    return np.mean(lst_auc_roc)
    
# Static parameters
cache_dir = "/tmp"
batch_size = 8

for ds_name in ['ds1', 'ds2', 'ds3']: # iterate over datasets ds1(random), ds2(No-Ngram), ds3(No-Class)

    A_members, A_nonmembers = read_data(ds_name+'.csv' if ds_name == 'ds3' else ds_name+'.arff')
    print('Dataset '+ds_name+' loaded!', len(A_members), len(A_nonmembers))

    ## Lauching code for MIA -> ROCAUC
    for model_name in ["EleutherAI/pythia-2.8b", "openlm-research/open_llama_3b_v2"]:
        print('Starting with model:', model_name)
        for single_metric in [True, False]: 
            print('Starting with single_metric:', single_metric)

            if single_metric:
                metric_list = ["10_min_probs", "ppl", "zlib_ratio", "10_max_probs"]
                for metric in metric_list:
                    print('Starting with metric:', metric)
                    get_mia_roc(A_members, A_nonmembers, model_name, cache_dir, batch_size, ds_name, metric)   
                    print('\n----------------------------------------------------------------------------\n\n')

            else:
                metric_list = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs"]
                get_mia_roc(A_members, A_nonmembers, model_name, cache_dir, batch_size, ds_name, metric_list)

    print('\n##############################################################################################################\n\n')