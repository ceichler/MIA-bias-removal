import pandas as pd


df = pd.read_csv('rest_proba_classification.csv')


def distance50(int):
    
    return abs(int-0.50)

# Apply preprocessing to the text column
df['dist'] = df['G score'].apply(distance50)

s_df = df.sort_values(by='dist', ascending=True)

# Filtering sorted df to produce tp, tn etc
# picking the 250 first
tp_df = s_df[(s_df['label'] == s_df['predicted class']) & (s_df['label'] == 'G')]
ptp_df = tp_df.head(250)
print('ptp df')
print(ptp_df.head)

fn_df = s_df[(s_df['label'] != s_df['predicted class']) & (s_df['label'] == 'G')]
pfn_df = fn_df.head(250)
print('pfn_df')
print(pfn_df.head)

tn_df = s_df[(s_df['label'] == s_df['predicted class']) & (s_df['label'] == 'G+')]
ptn_df = tn_df.head(250)
print('ptn df')
print(ptn_df.head)

fp_df = s_df[(s_df['label'] != s_df['predicted class']) & (s_df['label'] == 'G+')]
pfp_df = fp_df.head(250)
print('pfp df')
print(pfp_df.head)

# Concatenate the DataFrames vertically
concatenated_df = pd.concat([ptp_df, pfp_df, ptn_df, pfn_df], axis=0)

final_df = concatenated_df.drop(columns=['G score', 'G+ score', 'predicted class', 'dist'])
final_df.to_csv("ds3.csv", index=False)
