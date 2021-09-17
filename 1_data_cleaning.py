import pandas as pd
import numpy as np

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# For starters, we read the data and take a quick look

df1 = pd.read_csv("data_file.csv")
print(df1.head())
print(df1.shape)

# Okay, now we can start cleaning
# To begin, we group the 'area_type' category for unique value counts

print(df1.groupby('area_type')['area_type'].agg('count'))

# We'll assume not all columns are needed
# Make a new data frame with only the 'useful' data

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
print(df2.head())

# Now begins the data cleaning
# First step is to clear the NaN values
# We can find the total no. of NaN's in the frame, by column, with...

print(df2.isnull().sum())

# There are 73 missing values in the bathroom column
# You can drop these or fill them with a median value

df3 = df2.dropna()
print(df3.isnull().sum())

# Now we deal with irregularities
# For example, the 'size' column uses multiple phrases with the same meaning

print(df3['size'].unique())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.head())

# There will also be some general errors in the data
# For instance, take a look at the space requirements for homes with >20 bedrooms

print(df3[df3.bhk > 20])

# One house has 43 bedrooms but only takes up 2400 sq. ft.
# We need to explore 'total_sqft'

print(df3['total_sqft'].unique())

# There are ranges in there which can't be converted to numerical types
# There are also incorrect units
# We'll omit those and just worry about ranges
# Apply the convert_sqft_to_num function

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df5 = df4.dropna()
print(df5.head())
print(df5.shape)
print(df5.isnull().sum())

df5.to_csv('data_file_cleaned.csv', index = False)