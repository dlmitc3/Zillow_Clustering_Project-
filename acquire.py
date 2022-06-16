from env import host, user, password
import pandas as pd
import numpy as np
import os


################################### Get Connection to SQL Function ###################################

def get_connection(db, user = user, host = host, password = password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

###################################  Acquire New Zillow Data Function ###################################


def new_zillow_data():
    '''
    This function reads the zillow data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = '''
                select * 
                from properties_2017
                join predictions_2017 using (parcelid)
                left join airconditioningtype using (airconditioningtypeid)
                left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
                left join heatingorsystemtype using (heatingorsystemtypeid)
                left join propertylandusetype using (propertylandusetypeid)
                left join storytype using (storytypeid)
                left join typeconstructiontype using (typeconstructiontypeid)
                left join unique_properties using (parcelid)
                where latitude is not null 
                and longitude is not null
                and propertylandusetypeid = 261;
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_df.csv')
    return df


###################################  Get Zillow Data Function ###################################


def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from CodeUp database if cached == False 
    or if cached == True reads in mall customers df from a csv file, returns df.
    '''
    if cached or os.path.isfile('zillow_df.csv') == False:
        df = new_zillow_data()
    else:
        df = pd.read_csv('zillow_df.csv', index_col=0)
    return df


def new_iris_data():
    '''
    This function reads the iris data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM measurements AS m JOIN species USING (species_id)'
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris.csv')
    return df

def get_iris_data(cached=False):
    '''
    This function reads in iris data from CodeUp database if cached == False 
    or if cached == True reads in mall customers df from a csv file, returns df.
    '''
    if cached or os.path.isfile('iris.csv') == False:
        df = new_iris_data()
    else:
        df = pd.read_csv('iris.csv', index_col=0)
    return df