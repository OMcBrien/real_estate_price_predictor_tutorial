import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, sub_df in df.groupby('location'):
        m = np.mean(sub_df.price_per_sqft)
        st = np.std(sub_df.price_per_sqft)
        reduced_df = sub_df[(sub_df.price_per_sqft > (m-st)) & sub_df.price_per_sqft <= (m+st)]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)

    return df_out

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2 BHK', marker = 'o', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color='green', label='3 BHK', marker='+', s=50)
    plt.xlabel('Total sq. ft. area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {'mean': np.mean(bhk_df.price_per_sqft),
                                'std': np.std(bhk_df.price_per_sqft),
                                'count': bhk_df.shape[0]}
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')

df1 = pd.read_csv('data_file_cleaned_feateng.csv')

# For outlier removal, we want to rid the data of extreme/unphysical values
# We can use statistical techniques for this (e.g. standard deviation) or domain knowledge
# You could ask a business manager for a typical sq. ft. per bedroom here
# I'm going to pull some figures from Home Beautiful here:
# https://www.housebeautiful.com/uk/lifestyle/property/a35405209/average-house-price-england-square-foot-yes-homebuyers/
# UK average house area (sq. ft.) = 729
# UK house price per sq. ft. = £366
# I'll assume these are 2-3 bedroom houses so the sq. ft. price per bedroom should be £183 per bedroom per sq. ft.

print(df1.shape)

# First remove preposterously small places boasting high room counts
df2 = df1[~(df1.total_sqft / df1.bhk < 300)]
print(df2.shape)

# Now, price per sq. ft. (this is in GBP, but the prices are for Bangalore)
print(df2.price_per_sqft.describe())

plt.hist(df2.price_per_sqft, bins = 100, color = 'blue', alpha = 0.5)
plt.xlabel('Price per sq. ft.')

# A plot can help visualise this
# If the distribution is normal then ~66% of the distribution will be contained within 1 standard deviation of the mean
# We'll use a function to clip outliers based on location

df3 = remove_pps_outliers(df2)
plt.hist(df3.price_per_sqft, bins = 100, color = 'red', alpha = 0.5)
plt.show()

print(df2.shape, df3.shape)

# It may be a consequence of the conversion to GBP, but this step doesn't remove anything

# The last thing to worry about is the effect of localisation - some areas are inherently more valuable than others

plot_scatter_chart(df3, 'Hebbal')

# Try it for a few locations
# We're interested in properties where 2 bedrooms are priced higher than 3 bedrooms for approximately the same area
# Use the cleaning function for this outlier removal
# We create a dictionary with stats for the bedroom nos. and remove items from the n+1 bedroom properties who are lower in price than the n property mean

df4 = remove_bhk_outliers(df3)
print(df4.shape)
plot_scatter_chart(df4, 'Hebbal')

# For sanity, we plot a histogram...
plt.hist(df4.price_per_sqft, bins = 100, color = 'black', alpha = 0.5)
plt.show()

# What about property about there being so many bathrooms per unit area?
# Impose requirement that you can have 2 more bathrooms than you do rooms per property

df5 = df4[df4.bath < df4.bhk + 2]
print(df5.shape)

# Lastly, for model training, we are going to drop the data we don't actually need

df6 = df5.drop(['size', 'price_per_sqft'], axis = 'columns')
df6.to_csv('data_file_cleaned_feateng_outrem.csv', index = False)
