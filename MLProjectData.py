#Machine Learning Data Processing

'''
games.csv contains the raw data
fil_games.csv contains only needed columns
games_full_filter.csv contains only needed rows
    games_free_filter.csv filters out free games
games_prp.csv removes positive and negative in favor of positive review percentage
    games_free_prp.csv
games_prp_thresh.csv turns prp into binary 1 or 0, 1 if >= prp_threshold
    games_free_thresh.csv
games_norm.csv 1) divides metacritic score by 100, resulting in 0 < x < 1
               2) normalizes the log of minutes played
    games_free_norm.csv
rand_games.csv randomized the order of the entries
    rand_free_games.csv
games_tree.csv and games_free_tree.csv have the values grouped 0-10
    for better use of decision tree
'''
prp_threshold = 0.850

pos_thresh = 5
neg_thresh = 5
min_thresh = 10
max_thresh = 20000


import numpy as np
import pandas as pd
import math

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)

'''
games = pd.read_csv("games.csv")
#print(list(games.columns))
#print(games.head())

games_filtered = games.filter(['Metacritic score', 'Positive',
                               'Negative', 'Median playtime forever', 'Price'], axis=1)

print(games_filtered.head())
print(games_filtered.dtypes)

games_filtered.to_csv('fil_games.csv', index=False)
'''

'''
games = pd.read_csv("fil_games.csv")
games.columns = games.columns.str.replace(' ', '_')
print(games)

games = games.drop(games[games.Metacritic_score == 0].index)
print(games)

games = games.drop(games[games.Median_playtime_forever < min_thresh].index)
print(games)

games = games.drop(games[games.Median_playtime_forever > max_thresh].index)
print(games)

games = games.drop(games[games.Positive < pos_thresh].index)
print(games)

games = games.drop(games[games.Negative < neg_thresh].index)
print(games)

games = games.drop(games[games.Price < .01].index)
print(games)

print(games.describe())

games.to_csv('games_free_filter.csv', index=False)
'''

'''
games = pd.read_csv("games_free_filter.csv")
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)
print(games)

pos_perc = []
for i, row in games.iterrows():
    pos_perc.append(round(row['Positive'] / (row['Positive'] + row['Negative']), 3))

games.insert(4, "Positive_review_percentage", pos_perc, True)

games = games.filter(['Metacritic_score', 'Median_playtime_forever',
                      'Price', 'Positive_review_percentage'], axis=1)

print(games)
games.to_csv('games_free_prp.csv', index=False)
'''


'''
games = pd.read_csv("games_free_prp.csv")
print(games)


over_prp = []
for i, row in games.iterrows():
    if row['Positive_review_percentage'] < prp_threshold:
        over_prp.append(0)
    else:
        over_prp.append(1)

games.insert(3, "Positive_review_threshold", over_prp, True)

games = games.filter(['Metacritic_score', 'Median_playtime_forever',
                      'Price', 'Positive_review_threshold'], axis=1)

print(games)
games.to_csv('games_free_thresh.csv', index=False)
'''

'''
games = pd.read_csv("games_free_thresh.csv")
print(games)
print(games.describe())
print(games.sort_values(by=['Median_playtime_forever']))


meta_normal = []
for i, row in games.iterrows():
    meta_normal.append(round(row['Metacritic_score'] / 100, 2))

games = games.filter(['Median_playtime_forever', 'Price', 'Positive_review_threshold'], axis=1)
games.insert(0, 'Metacritic_score', meta_normal, True)
print(games)

time_min = math.log(games['Median_playtime_forever'].min())
time_max = math.log(games['Median_playtime_forever'].max())
time_dif = time_max - time_min

time_normal = []
for i, row in games.iterrows():
    time_normal.append(round((math.log(row['Median_playtime_forever']) - time_min) / time_dif, 3))

games = games.filter(['Metacritic_score', 'Price', 'Positive_review_threshold'], axis=1)
games.insert(1, 'Median_playtime', time_normal, True)
print(games)

price_min = games['Price'].min()
price_max = games['Price'].max()
price_dif = price_max - price_min

price_normal = []
for i, row in games.iterrows():
    price_normal.append(round((row['Price'] - price_min) / price_dif, 3))

games = games.filter(['Metacritic_score', 'Median_playtime', 'Positive_review_threshold'], axis=1)
games.insert(2, 'Price', price_normal, True)
print(games)

print(games.describe())

games.to_csv('games_free_norm.csv', index=False)
'''

'''
games = pd.read_csv("games_free_norm.csv")
print(games)

games = games.sample(frac=1)
print(games)

games.to_csv('rand_free_games.csv', index=False)
'''

#games = pd.read_csv("games_prp_thresh.csv")
#print(games.describe())

'''
games = pd.read_csv("rand_free_games.csv")
meta_scale = []
for i, row in games.iterrows():
    meta_scale.append(int(row['Metacritic_score'] * 10))

games = games.filter(['Median_playtime', 'Price', 'Positive_review_threshold'], axis=1)
games.insert(0, 'Metacritic_score', meta_scale, True)
print(games)

time_scale = []
for i, row in games.iterrows():
    time_scale.append(int(row['Median_playtime'] * 10))

games = games.filter(['Metacritic_score', 'Price', 'Positive_review_threshold'], axis=1)
games.insert(1, 'Median_playtime', time_scale, True)
print(games)

price_scale = []
for i, row in games.iterrows():
    price_scale.append(int(row['Price'] * 10))

games = games.filter(['Metacritic_score', 'Median_playtime', 'Positive_review_threshold'], axis=1)
games.insert(2, 'Price', price_scale, True)
print(games)

print(games.describe())

games.to_csv('games_free_tree.csv', index=False)
'''



