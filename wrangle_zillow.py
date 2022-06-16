import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from acquire import get_zillow_data



def wrangle_zillow(cached=True):
    '''
    This function prepares the data for exploration by 
    handling null values and outliers,
    creating new features from existing features,
    and splitting the data into train, validate and test
    '''
    #get zillow data
    df = get_zillow_data()
    #filling nulls with appropriate values
    df = fill_nulls(df)
    #removing outliers from data
    df = remove_outliers(df)
    #creating new features
    df = create_features(df)
    #filter out columns and rows with more than 40% null values
    df = handle_missing_values(df, .6, .6)
    #dropping unitcnt since they are all the same and unnecessary columns
    df = df.drop(columns = ['propertylandusetypeid', 'propertycountylandusecode', 'propertylandusedesc',
                             'calculatedbathnbr', 'finishedsquarefeet12', 'heatingorsystemtypeid', 
                            'id', 'fips', 'fullbathcnt', 'propertyzoningdesc', 'unitcnt',
                            'regionidcounty', 'id.1', 'assessmentyear', 
                            'censustractandblock', 'rawcensustractandblock', 'buildingqualitytypeid'])
    #drop all rows with missing values
    df.dropna(inplace = True)
    #split into train, validate, test
    train, validate, test = zillow_split(df)
    return train, validate, test



def fill_nulls(df):
    '''
    This function fill nulls with appropriate values
    '''
    df.poolcnt = df.poolcnt.fillna(0)
    df.fireplacecnt = df.fireplacecnt.fillna(0)
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('None')
    df.unitcnt = df.unitcnt.fillna(1)
    return df

def remove_outliers(df):
    '''
    This function removes outliers and 
    '''
    #filter out bedrooms and bathrooms == 0
    df = df[(df.bedroomcnt > 0) & (df.bedroomcnt <= 7) & (df.bathroomcnt > 0) & (df.bathroomcnt <= 7)]
    #filter out houses less than 400 square feet
    df = df[(df.calculatedfinishedsquarefeet > 400) & (df.calculatedfinishedsquarefeet < 7000)]
    #filter out all units not equal to 1
    df = df[df.unitcnt == 1]
    #removing heating or system source outliers
    df = df[~df.heatingorsystemdesc.isin(['Yes', 'Gravity', 'Radiant', 'Baseboard', 'Solar', 'Forced air'])]
    return df


def create_features(df):
    '''
    This functions creates new features that are more 
    apllicable and familiar out of existing features
    '''
    df['age'] = 2017 - df.yearbuilt
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt
    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    # ratio of beds to baths
    df['bed_bath_ratio'] = df.bedroomcnt/df.bathroomcnt
    #changing numbered labels into appropriate names
    df['county'] = df.fips.replace([6037, 6059, 6111],['los_angeles', 'orange', 'ventura'])
    #changing names of heating system
    df.heatingorsystemdesc = df.heatingorsystemdesc.replace(['Central', 'Floor/Wall', 'None'], ['central_heating', 'floor_wall_heating', 'no_heating'])
    #creating dummy variables
    county_df = pd.get_dummies(df.county)
    heating_or_system_df = pd.get_dummies(df.heatingorsystemdesc)
    #adding dummies back into main dataframe
    df = pd.concat([df, county_df, heating_or_system_df], axis=1)
    #duplicating logerror so it will be at the end of the list
    df['error'] = df.logerror
    #filter out outliers on new features
    df = df[(df.acres < 10) & (df.taxrate < .05)]
    #drop duplicate columns
    df = df.drop(columns = ['bathroomcnt', 'taxamount', 'taxvaluedollarcnt', 
                       'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                       'yearbuilt', 'lotsizesquarefeet', 'logerror'])
    return df


def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    This function takes in a Dataframe, 
    proportion(0-1) of nulls required for a column
    and a proportion(0-1) of nulls required for rows
    then returns a dataframe without the nulls 
    under the threshold
    '''
    #setting threshold for row, only accepts integer
    thresh_row = int(round(prop_required_column*df.shape[0],0))
    #dropping nulls under threshold
    df.dropna(axis=1, thresh=thresh_row, inplace=True)
    #setting threshold for columns, only accepts integer
    thresh_col = int(round(prop_required_row*df.shape[1],0))
    #dropping nulls under threshold
    df.dropna(axis=0, thresh=thresh_col, inplace=True)
    return df 



def zillow_split(df):
    '''
    This function splits a dataframe into train, validate, and test sets
    '''
    train_and_validate, test = train_test_split(df, train_size=.8, random_state=123)
    train, validate = train_test_split(train_and_validate, train_size = .7, random_state=123)
    return train, validate, test

def scaled_zillow_columns(cached = True):
    '''
    This function uses a MinMaxScaler to scale numeric columns
    from the wrangle_zillow function
    '''
    train, validate, test = wrangle_zillow()
    columns_to_scale= ['bedroomcnt', 'calculatedfinishedsquarefeet', 'fireplacecnt', 'latitude', 'longitude', 'poolcnt', 'regionidcity', 'regionidzip', 'roomcnt', 'age', 'taxrate', 'acres', 'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'bed_bath_ratio']
    #initialize scaler function
    scaler = sklearn.preprocessing.MinMaxScaler()
    #adds '_scaled' to columns that will be scaled
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    #fitting columns to be scaled
    scaler.fit(train[columns_to_scale])
    #adding scaled columns back into their respective dataframes
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    train = train.drop(columns = columns_to_scale)
    validate= validate.drop(columns = columns_to_scale)
    test = test.drop(columns = columns_to_scale)
    
    return train, validate, test


################################# Null Finder Functions #################################

def null_finder_columns(df):
    '''
    This function takes in a DataFrame and list 
    information about the null values in the columns
    '''
    #accepts a 'df' and creates a new one labeled 'nulls'  
    #nulls index is the df's columns
    nulls = pd.DataFrame(index = df.columns)
    #sums up the null values in the dataframes columns
    nulls['num_rows_missing'] = df.isnull().sum(axis = 0)
    #finds the percentage of null values in the df's columns
    nulls['pct_rows_missing'] = nulls.num_rows_missing / df.shape[0]
    return nulls



def null_finder_rows(df):
    '''
    This function finds the number of columns missing in a row,
    the percent of columns missing in the row
    and the number of rows that have the same amount of columns missing
    '''
    #initiate a dataframe
    rows = pd.DataFrame()
    #find the number of columns missing in the row
    rows['num_cols_missing'] = df.isnull().sum(axis=1)
    #find the percentage of columns missing in the row
    rows['pct_cols_missing'] = df.isnull().sum(axis=1) / df.shape[1]
    #group by 'num_cols_missing' and find 
    #how many rows have that number of columns missing
    num_rows = rows.groupby('num_cols_missing').count()
    #rename the column as 'num_rows'
    num_rows = num_rows.rename(columns ={'pct_cols_missing': "num_rows"})
    #group by 'num_cols_missing' and find 
    #the percentage of columns missing in the row
    pct_cols = rows.groupby('num_cols_missing').mean()
    #combine the 'pct_cols' and 'num_rows'
    result = pd.concat([pct_cols, num_rows], axis=1, sort=False)
    #take the 'num_cols_missing' out of the index
    result = result.reset_index()
    return result