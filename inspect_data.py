import pandas as pd

files = ['stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv']
for f in files:
    df = pd.read_csv(f'data/{f}', nrows=3)
    print(f'\n=== {f} ===')
    print(df.dtypes.to_string())
    print(df.to_string())

print('\n=== train.csv (first 5 rows) ===')
df_train = pd.read_csv('data/train.csv', nrows=5)
print(df_train.dtypes.to_string())
print(df_train.to_string())
print('Families:', pd.read_csv('data/train.csv', usecols=['family'])['family'].nunique())
print('Stores:', pd.read_csv('data/train.csv', usecols=['store_nbr'])['store_nbr'].nunique())
dates = pd.read_csv('data/train.csv', usecols=['date'])
print('Date range:', dates['date'].min(), 'to', dates['date'].max())
