import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load your dataset
file_path = 'amr_horse_ds.csv'  # Change this to the path of your CSV file
horse_df = pd.read_csv(file_path)

# Data preprocessing
# Binning 'Age' column (if exists in your dataset) and other preprocessing steps
# You should modify these steps based on your actual data and preprocessing needs

# Convert boolean columns to 'Yes'/'No' for association rule mining
for col in horse_df.columns[horse_df.dtypes == 'bool']:
    horse_df[col] = horse_df[col].map({True: 'Yes', False: 'No'})

# Convert the DataFrame into a list of transactions
transactions = horse_df.apply(lambda row: [str(row[col]) for col in horse_df.columns if row[col] != 'No'],
                              axis=1).tolist()

# Transaction encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Parameters for association rule mining
min_sup = [0.05, 0.1, 0.4]
min_conf = [0.70, 0.85, 0.95]
min_lift = [1.1, 1.5, 4]

# Extracting rules
best_params = None
best_rules = None
best_rules_count = 0

for support in min_sup:
    for confidence in min_conf:
        for lift in min_lift:
            frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
            rules = rules[rules['lift'] >= lift]

            if 20 <= len(rules) <= 50:
                if not best_params or len(rules) > best_rules_count:
                    best_params = (support, confidence, lift)
                    best_rules = rules
                    best_rules_count = len(rules)

# Output the results
print("Best parameters:", best_params)
print("Number of rules:", best_rules_count)
if best_rules is not None:
    print(best_rules)
else:
    print("No suitable parameter set found.")
