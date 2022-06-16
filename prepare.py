import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
random_state = 42


def prep_zillow(df):
    '''
    This function takes in the zillow df acquired from the codeup MySQL database and performs 
    the following actions:
    - filters the data for only Single Family Residential properties
    - drops redundant foreign key identification code columns
    - drops other redundant, irrelevant, or non-useful columns
    - handles null valuesby:
        - filling null values with 0's in the following columns, since it is reasonable to assume nulls in these columns represent zero values: 
            - `fireplacecnt`, `garagecarcnt`, `garagetotalsqft`, `hashottuborspa`, `poolcnt`, `threequarterbathnbr`, `taxdelinquencyflag`
        - then dropping columns that remain where greater than 5% of values in that column are null
        - then dropping rows that remain with any null values
    - changes data types to more appropriately reflect the data they represent
    - adds the following engineered feature columns (see data-dictionary for details):
        - `age`, `bool_has_garage`, `bool_has_pool`, `bool_has_fireplace`, `taxvalue_per_sqft`, `taxvalue_per_bedroom`, `taxvalue_per_bathroom`
    - adds the following target-related columns (for exploration) (see data-dictionary for details): 
        - `abs_logerror`, `logerror_direction`
    The cleaned and prepped df is returned.
    '''
    # drop redundant id code columns
    id_cols = [col for col in df.columns if 'typeid' in col or col in ['id', 'parcelid']]
    df = df.drop(columns=id_cols)
    # filter for single family properties
    df = df[df.propertylandusedesc == 'Single Family Residential']
    # drop specified columns
    cols_to_drop = ['calculatedbathnbr',
                    'finishedfloor1squarefeet',
                    'finishedsquarefeet12', 
                    'regionidcity',
                    'landtaxvaluedollarcnt',
                    'taxamount',
                    'rawcensustractandblock',
                    'roomcnt',
                    'regionidcounty']
    df = df.drop(columns=cols_to_drop)
    # fill null values with 0 in specified columns
    cols_to_fill_zero = ['fireplacecnt',
                         'garagecarcnt',
                         'garagetotalsqft',
                         'hashottuborspa',
                         'poolcnt',
                         'threequarterbathnbr',
                         'taxdelinquencyflag']
    for col in cols_to_fill_zero:
        df[col] = np.where(df[col].isna(), 0, df[col]) 
    # drop columns with more than 5% null values
    for col in df.columns:
        if df[col].isnull().mean() > .05:
            df = df.drop(columns=col)
    # drop rows that remain with null values
    df = df.dropna()   
    # changing numeric codes to strings
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['regionidzip'] = df.regionidzip.apply(lambda x: str(int(x)))
    df['censustractandblock'] = df.censustractandblock.apply(lambda x: str(int(x)))
    # change the 'Y' in taxdelinquencyflag to 1
    df['taxdelinquencyflag'] = np.where(df.taxdelinquencyflag == 'Y', 1, df.taxdelinquencyflag)
    # change boolean column to int
    df['hashottuborspa'] = df.hashottuborspa.apply(lambda x: str(int(x)))
    # changing year from float to int
    df['yearbuilt'] = df.yearbuilt.apply(lambda x: int(x))
    df['assessmentyear'] = df.yearbuilt.apply(lambda x: int(x))
    # moving the latitude and longitude decimal place
    df['latitude'] = df.latitude / 1_000_000
    df['longitude'] = df.longitude / 1_000_000
    # adding a feature: age 
    df['age'] = 2017 - df.yearbuilt
    # add a feature: has_garage
    df['bool_has_garage'] = np.where(df.garagecarcnt > 0, 1, 0)
    # add a feature: has_pool
    df['bool_has_pool'] = np.where(df.poolcnt > 0, 1, 0)
    # add a feature: has_fireplace
    df['bool_has_fireplace'] = np.where(df.fireplacecnt > 0, 1, 0)
    # add a feature: taxvalue_per_sqft
    df['taxvalue_per_sqft'] = df.taxvaluedollarcnt / df.calculatedfinishedsquarefeet
    # add a feature: taxvalue_per_bedroom
    df['taxvalue_per_bedroom'] = df.taxvaluedollarcnt / df.bedroomcnt
    #add a feature: taxvalue_per_bathroom
    df['taxvalue_per_bathroom'] = df.taxvaluedollarcnt / df.bathroomcnt    
    #add a feature: taxvalue_per_room
    df['taxvalue_per_bathroom'] = df.taxvaluedollarcnt / (df.bathroomcnt + df.bedroomcnt)
    # adding prefix to boolean columns
    df = df.rename(columns={'hashottuborspa': 'bool_hashottuborspa'})
    df = df.rename(columns={'taxdelinquencyflag': 'bool_taxdelinquencyflag'})
    # rename sqft column
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sqft'})
    # add a column: absolute value of logerror (derived form target)
    df['abs_logerror'] = abs(df.logerror)
    # add a column: direction of logerror (high or low) (derived from target)
    df['logerror_direction'] = np.where(df.logerror < 0, 'low', 'high')


    return df

def train_validate_test_split(df, test_size=.2, validate_size=.3, random_state=random_state):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    # split the dataframe into train and test
    train, test = train_test_split(df, test_size=.2, random_state=random_state)
    # further split the train dataframe into train and validate
    train, validate = train_test_split(train, test_size=.3, random_state=random_state)
    # print the sample size of each resulting dataframe
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test

def remove_outliers(train, validate, test, k, col_list):
    ''' 
    This function takes in a dataset split into three sample dataframes: train, validate and test.
    It calculates an outlier range based on a given value for k, using the interquartile range 
    from the train sample. It then applies that outlier range to each of the three samples, removing
    outliers from a given list of feature columns. The train, validate, and test dataframes 
    are returned, in that order. 
    '''
    # Create a column that will label our rows as containing an outlier value or not
    train['outlier'] = False
    validate['outlier'] = False
    test['outlier'] = False
    for col in col_list:

        q1, q3 = train[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        train['outlier'] = np.where(((train[col] < lower_bound) | (train[col] > upper_bound)) & (train.outlier == False), True, train.outlier)
        validate['outlier'] = np.where(((validate[col] < lower_bound) | (validate[col] > upper_bound)) & (validate.outlier == False), True, validate.outlier)
        test['outlier'] = np.where(((test[col] < lower_bound) | (test[col] > upper_bound)) & (test.outlier == False), True, test.outlier)

    # remove observations with the outlier label in each of the three samples
    train = train[train.outlier == False]
    train = train.drop(columns=['outlier'])

    validate = validate[validate.outlier == False]
    validate = validate.drop(columns=['outlier'])

    test = test[test.outlier == False]
    test = test.drop(columns=['outlier'])

    # print the remaining 
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test

def scale_zillow(train, validate, test, target, scaler_type=MinMaxScaler()):
    '''
    This takes in the train, validate, and test dataframes, as well as the target label. 
    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the train, validate, and test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 
    train, validate, and test dataframes are returned, in that order. 
    '''
    # identify quantitative features to scale
    quant_features = [col for col in train.columns if (train[col].dtype != 'object') 
                                                    & (target not in col) 
                                                    & ('bool_' not in col)]
    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)
    # screate and fit the scaler
    scaler = scaler_type.fit(train[quant_features])
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
    # add 'scaled' prefix to columns
    for feature in quant_features:
        train_scaled = train_scaled.rename(columns={feature: f'scaled_{feature}'})
        validate_scaled = validate_scaled.rename(columns={feature: f'scaled_{feature}'})
        test_scaled = test_scaled.rename(columns={feature: f'scaled_{feature}'})
    # concat scaled feature columns to original train, validate, test df's
    train = pd.concat([train, train_scaled], axis=1)
    validate = pd.concat([validate, validate_scaled], axis=1)
    test = pd.concat([test, test_scaled], axis=1)

    return train, validate, test

def encode_zillow(train, validate, test, target):
    '''
    This function takes in the train, validate, and test samples, as well as a label for the target variable. 
    It then encodes each of the categorical variables using one-hot encoding with dummy variables and appends 
    the new encoded variables to the original dataframes as new columns with the prefix 'enc_{variable_name}'.
    train, validate and test dataframes are returned (in that order)
    '''
    # identify the features to encode (categorical features represented by non-numeric data types)
    features_to_encode = [col for col in train.columns if (train[col].dtype == 'object')
                                                        & ('bool_' not in col) 
                                                        & (target not in col)
                                                        & (train[col].nunique() < 25)]
    #iterate through the list of features                  
    for feature in features_to_encode:
        # establish dummy variables
        dummy_df = pd.get_dummies(train[feature],
                                  prefix=f'enc_{train[feature].name}',
                                  drop_first=True)
        # add the dummies as new columns to the original dataframe
        train = pd.concat([train, dummy_df], axis=1)

    # then repeat the process for the other two samples:

    for feature in features_to_encode:
        dummy_df = pd.get_dummies(validate[feature],
                                  prefix=f'enc_{validate[feature].name}',
                                  drop_first=True)
        validate = pd.concat([validate, dummy_df], axis=1)
        
    for feature in features_to_encode:
        dummy_df = pd.get_dummies(test[feature],
                                  prefix=f'enc_{test[feature].name}',
                                  drop_first=True)
        test = pd.concat([test, dummy_df], axis=1)
    
    return train, validate, test

def add_clusters(train, validate, test):
    '''
    This function takes in the train, validate, and test samples from the zillow dataset.
    It then performs clustering on various combinations of features in the train sample, 
    using the process outlined in the explore_notebook.ipynb. 
    Those clusters are then given useful names where appropriate, and added
    as categorical features to the dataset.
    The train, validate, and test df's are returned, in that order.
    '''
    
    # cluster_BedBath

    # identify features
    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt']
    # create the df to cluster on 
    x = train[features]
    # create and fit the KMeans object
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    # create cluster labels for each of the samples and add as an additional column
    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBath'] = kmeans.predict(x)
        sample['cluster_BedBath'] = sample.cluster_BedBath.map({1:'low', 0:'mid', 2:'high'})

    # repeat the process for each of the desired feature combinations on which to cluster

    # cluster_BedBathSqft

    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt', 'scaled_sqft']
    x = train[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBathSqft'] = kmeans.predict(x)
        sample['cluster_BedBathSqft'] = sample.cluster_BedBathSqft.map({1:'low', 0:'mid', 2:'high'})

    # cluster_LatLong
    features = ['scaled_latitude', 'scaled_longitude']
    x = train[features]
    kmeans = KMeans(n_clusters=4, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_LatLong'] = kmeans.predict(x)
        sample['cluster_LatLong'] = sample.cluster_LatLong.map({0:'east', 1:'central', 2:'west', 3:'north'})

    # cluster_BedBathTaxvaluepersqft
    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt', 'scaled_taxvalue_per_sqft']
    x = train[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBathTaxvaluepersqft'] = kmeans.predict(x)
        sample['cluster_BedBathTaxvaluepersqft'] = sample.cluster_BedBathTaxvaluepersqft.astype(str)

    return train, validate, test