import uuid
import pprint

import pandas
import requests
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Read in a csv file from a given path
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

#plots Heat map for feature correlation
def heatMap(dataframe):
    plt.figure(figsize=(30, 30))
    sns.heatmap(dataframe.corr(), annot=True, linewidths=0.01, cmap='coolwarm', cbar=True)
    plt.show()

    print("Skewness")
    print(dataframe.skew(axis=0, skipna=True))

    print("kurtosis")
    print(dataframe.kurtosis())



def read_CSV(filepath):
    df = pandas.read_csv(filepath)

    class_dist = df.groupby('beds').size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.show()

    class_dist = df.groupby('guests_included').size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.show()

    class_dist = df.groupby('room_type').size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.show()

    class_dist = df.groupby('accommodates').size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.show()

    heatMap(df)
    return df


# Replace all nan values in a column with a value or statistic
def nan_removal(df, col, is_percent=False, is_categorical=False):
    if df[col].isnull().values.any():
        # print("Removing NaNs from: ", col)
        if is_percent:
            df[col] = df[col].apply(percent_to_number)
            # print(df[col].mean())
            df[col].fillna(df[col].mean(), inplace=True)
        if is_categorical:
            # print(df[col].mode()[0])
            df[col].fillna(df[col].mode()[0], inplace=True)


# prints columns that contain NaNs
def nan_checker(df):
    print('Checking NaNs...')
    count = 0
    for col in list(df):
        if df[col].isnull().values.any():
            count += 1
            print("{0} has NaNs: ".format(col))
    if not count:
        print("Columns are NaN free.\n")


# Convert string to integer and return the integer
def percent_to_number(x):
    if isinstance(x, str):
        return int(x[:-1])
    else:
        return x


# Counts the length of a list in a column a creates a new column with the count
def count_list_in_column(df, col, new_name):
    df[new_name] = df[col].apply(lambda x: len(x.split(",")))
    del (df[col])


# Converts a column that contains a list to multiple columns with one-hot encoding
def column_converter(df, col):
    values = get_distinct_values(df, col)
    if '' in values:
        values.remove('')
    for name in values:
        create_column(df, name, 0)
    for index, row in df.iterrows():
        columns = to_list(row[col])
        set_values(df, columns, index)
    del (df[col])


# Set the columns to 1 that are present in the root column
def set_values(df, columns, index):
    if '' in columns: columns.remove('')
    for column in columns:
        df.at[index, column] = 1


# Creates a new column with initial value in the dataframe
def create_column(df, name, initial_value):
    df[name] = initial_value


# Creates a set of unique value that appear within one column
def get_distinct_values(df, col):
    distinct_values = set()
    values = df[col].apply(to_list)
    for v in values:
        distinct_values |= set(v)
    return distinct_values


# encode categorical features (columns of strings to columns of integers)
def encode(df, col):
    keys = {x: i for i, x in enumerate(list(set(df[col])))}
    # print(keys)
    df[col] = df[col].map(keys)


# Converts the string "{'a','b','c'}" to a real python list
def to_list(x):
    return x[1:-1].lower().split(",")


# Converts the price column to an integer
def convert_price_to_integer(df, col):
    df[col] = df[col].apply(lambda x: float(x.replace('$', '').replace(',', '').replace('"', '')))

# encode categorical features (columns of strings to columns of integers)
def encode(df, col):
    keys = {x: i for i, x in enumerate(list(set(df[col])))}
    # print(keys)
    df[col] = df[col].map(keys)


# Shuffles the rows of the given file and stores it back on disk
def shuffle_file(filename):
    df = pandas.read_csv(filename)
    df = df.sample(frac=1)
    df.to_csv(filename, index=False)


# Reduces the number of rows of the given file and stores it into anotehr file
def reduce_size_to(filename, rows, newfile):
    df = pandas.read_csv(filename)
    df = df.head(rows)
    df.to_csv(newfile, index=False)


# Add a new column to the given file with default value
def add_column_with(file, col, value):
    df = pandas.read_csv(file)
    df[col] = value
    df.to_csv(file)


# Concatenates all files in the given list
def csv_concat(filelist):
    df = pandas.DataFrame()
    for files in filelist:
        data = pandas.read_csv(files)
        df = df.append(data, sort=False)
    df.reset_index().drop(
        ['bed_type', 'neighbourhood', 'cancellation_policy', 'description', 'index', 'Unnamed: 0', 'host_since', 'host_name', 'host_id', 'id', 'market'],
        axis=1).to_csv('./listings_first_preproccesed.csv', index=False)

def LabelEncoderAlgo(dataframe,col):
    le = preprocessing.LabelEncoder()
    le.fit(dataframe[col])
    dataframe[col]=le.transform(dataframe[col])
    return dataframe

def OneHotEncoderAlgo(dataframe,col):
    dataframe_d1 = pd.DataFrame()
    dataframe_d1['room_type'] = dataframe['room_type']
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(dataframe_d1)
    x = enc.transform(dataframe_d1).toarray()
    for j in range(len(enc.get_feature_names())):
        dataframe[enc.get_feature_names()[j]] = [i[j] for i in x]

    dataframe = dataframe.drop(columns=['room_type'])
    return dataframe

def Normalization(dataframe):
    indexNames = dataframe[dataframe['price'] > 1000].index
    dataframe.drop(indexNames, inplace=True)



    scalar = MinMaxScaler()
    scalar.fit(dataframe)
    dataframe = pd.DataFrame(scalar.transform(dataframe),columns=dataframe.columns)
    dataframe['price'] = dataframe['price'].apply(lambda x: x * 1000)
    return dataframe

# function that returns cleaned dataframe
def get_processed_data(file):
    df = read_CSV(file)

    nan_checker(df)
    try:
        nan_removal(df, 'host_acceptance_rate', is_percent=True)
        nan_removal(df, 'host_response_time', is_categorical=True)
        nan_removal(df, 'host_response_rate', is_percent=True)
        nan_removal(df, 'beds', is_categorical=True)
        nan_removal(df, 'review_scores_rating', is_percent=True)
    except Exception as e:
        print(e)

    convert_price_to_integer(df, 'price')
    column_converter(df, 'amenities')
    count_list_in_column(df, 'host_verifications', "verifications_count")

    df = LabelEncoderAlgo(df, 'host_identity_verified')
    df = LabelEncoderAlgo(df, 'host_response_time')
    df = LabelEncoderAlgo(df, 'host_is_superhost')
    df = LabelEncoderAlgo(df, 'property_type')
    df = OneHotEncoderAlgo(df, 'room_type')

    df = Normalization(df)

    df = df.round(2)
    df = df.dropna()


    df.to_csv('./listings_preprocessed.csv', index=False)
    return df


csv_concat(['./listings_details_Newyork.csv', './listings_details_Boston.csv', './listings_details_Seattle.csv'])
get_processed_data('./listings_first_preproccesed.csv')


