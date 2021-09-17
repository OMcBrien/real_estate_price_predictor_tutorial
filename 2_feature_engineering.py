import pandas as pd

df = pd.read_csv('data_file_cleaned.csv')
print(df.head())

# Introduce new value called 'price_per_sqft'
# Bear in mind price is in Lahk Rupees
# The conversion to GBP is approx. £989.12 per Lahk Rupee
# We'll keep it in rupees for now

df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
print(df.head())

# Now we explore the 'location' feature

print(len(df.location.unique()))

# 1304 unique locations is a dimensionality curse, we have too much data for one hot encoding to be useful
# We'll strip some of the extra spaces in the values

df.location.apply(lambda x: x.strip())
print(df.head())
location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending = False)
print(location_stats)

# Now, there's a lot of values that only appear once
# How about a rule where if a location appears fewer than 10 times, we call it 'other'?

location_stats_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df.location.unique()))

df.to_csv('data_file_cleaned_feateng.csv', index = False)

