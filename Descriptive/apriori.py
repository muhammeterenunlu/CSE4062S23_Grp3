import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import os

def apriori_algorithm():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load your data
    data = pd.read_excel("preprocessed_data.xlsx")

    # Each row will be considered as a transaction, but as a list of column names where the cell value is True
    transactions = [[data.columns[j] for j in range(len(data.columns)) if row[j]] for i, row in data.iterrows()]

    # Use TransactionEncoder to transform the data
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)

    # Create a new dataframe
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Run apriori algorithm on the dataframe, with minimum support threshold as 0.6
    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

    # Sort the dataframe in descending order by the 'support' column
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

    # Save to JSON
    frequent_itemsets.to_json("frequent_itemsets_apriori.json", orient='records', lines=True)

    # Set the maximum column width to a large number to avoid truncation
    pd.set_option('display.max_colwidth', 1000)

    print("Apriori Algorithm Results:")
    print(frequent_itemsets)
