import pandas as pd

#import the csv dataset into a pandas dataframe
data = pd.read_csv('amazon-cellphones.csv')

#drop the unverified comments
for x in data.index:
    if data.loc[x, "verified"] == "false":
        data.drop(x, inplace=True)

#get the verified comments as a list
reviews = data['body'].tolist()

