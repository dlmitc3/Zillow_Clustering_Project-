import pandas as pd
import sklearn as sk
from math import sqrt
from sklearn.linear_model import LinearRegression, LassoLars 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def display_model_results(model_results, as_std_from_baseline=False):
    '''
    This function takes in the model_results dataframe created in the Model stage of the 
    project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of
    models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    if as_std_from_baseline:
        return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='std_from_baseline',
                                     aggfunc=lambda x: x)
    else:
        return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)        

def get_best_model_results(model_results, n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.
    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 
    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores
    best_models = (model_results[(model_results.sample_type == 'validate')]
                                                 # sort by score value in ascending order
                                                 .sort_values(by='score', 
                                                              ascending=True)
                                                 # take only the model number for the top n_models
                                                 .head(n_models).model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results

def determine_regression_baseline(train, target):
    '''
    This function takes in a train sample and a continuous target variable label and 
    determines whether the mean or median performs better as a baseline prediction. 
    '''
    # create empty dataframe for storing prediction results
    results = pd.DataFrame(index=train.index)
    # assign actual values for the target variable
    results['actual'] = train[target]
    # assign a baseline using mean
    results['baseline_mean'] = train[target].mean()
    # assign a baseline using median
    results['baseline_median']= train[target].median()
    
    # get RMSE values for each potential baseline
    RMSE_baseline_mean = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_mean))
    RMSE_baseline_median = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_median))
    
    # compare the two RMSE values; drop the lowest performer and assign the highest performer to baseline variable
    if RMSE_baseline_median < RMSE_baseline_mean:
        results = results.drop(columns='baseline_mean')
        results['RMSE_baseline'] = RMSE_baseline_median
        baseline_type = 'median'
    else:
        results = results.drop(columns='baseline_median')
        results['RMSE_baseline'] = RMSE_baseline_mean
        baseline_type = 'mean'
    
    return baseline_type

def run_baseline(train,
                 validate,
                 target,
                 model_number,
                 model_info,
                 model_results):
    
    baseline_type = determine_regression_baseline(train, target)

    y_train = train[target]
    y_validate = validate[target]

    # identify model number
    model_number = 'baseline'
    #identify model type
    model_type = 'baseline'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)


    # establish baseline predictions for train sample
    y_pred = baseline_pred = pd.Series(train[target].mean()).repeat(len(train))

    # get metrics
    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)


    # establish baseline predictions for validate sample
    if baseline_type == 'mean':
        y_pred = baseline_pred = pd.Series(validate[target].mean()).repeat(len(validate))
    elif baseline_type == 'median':
        y_pred = baseline_pred = pd.Series(validate[target].median()).repeat(len(validate))

    # get metrics
    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)
    
    model_number = 0
    
    return model_number, model_info, model_results

def run_OLS(train, validate, target, model_number, model_info, model_results):
    # including the most highly correlated features from exploration
    features = ['scaled_sqft',
                'scaled_bedroomcnt',
                'scaled_bathroomcnt',
                'scaled_fullbathcnt',
                'scaled_age',
                'scaled_assessmentyear',
                'scaled_yearbuilt',
                'scaled_garagecarcnt',
                'scaled_garagetotalsqft']

    # establish model number
    model_number += 1

    # establish model type
    model_type = 'OLS regression'

    # create a dictionary containing the features and hyperparamters used in this model instance
    dct = {'model_number': model_number,
           'model_type': model_type,
           'features': features}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)

    #split the samples into x and y
    x_train = train[features]
    y_train = train[target]

    x_validate = validate[features]
    y_validate = validate[target]

    # create the model object and fit to the training sample
    linreg = LinearRegression(normalize=True).fit(x_train, y_train)

    # make predictions for the training sample
    y_pred = linreg.predict(x_train)
    sample_type = 'train'

    # store information about model performance
    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'sample_type': sample_type, 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)

    # make predictions for the validate sample
    y_pred = linreg.predict(x_validate)
    sample_type = 'validate'

    # store information about model performance
    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'sample_type': sample_type, 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)
    
    return model_number, model_info, model_results

def run_PolyReg(train, validate, target, model_number, model_info, model_results):
    
    for degree in range(2, 5):

        # including the most highly correlated features from exploration
        features = ['scaled_sqft',
                    'scaled_bedroomcnt',
                    'scaled_bathroomcnt',
                    'scaled_fullbathcnt',
                    'scaled_age',
                    'scaled_assessmentyear',
                    'scaled_yearbuilt',
                    'scaled_garagecarcnt',
                    'scaled_garagetotalsqft']

        # establish model number
        model_number += 1

        # establish model type
        model_type = 'polynomial regression'

        # create a dictionary containing the features and hyperparamters used in this model instance
        dct = {'model_number': model_number,
            'model_type': model_type,
            'features': features,
            'degree': degree}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create a polynomial features object
        pf = PolynomialFeatures(degree=degree)
        
        # fit and transform the data
        x_train_poly = pf.fit_transform(x_train)
        x_validate_poly = pf.fit_transform(x_validate)

        # create the model object and fit to the training sample
        linreg = LinearRegression().fit(x_train_poly, y_train)
        
        # make predictions for the training sample
        y_pred = linreg.predict(x_train_poly)
        sample_type = 'train'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
            'sample_type': sample_type, 
            'metric_type': 'RMSE',
            'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)

        # make predictions for the validate sample
        y_pred = linreg.predict(x_validate_poly)
        sample_type = 'validate'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
            'sample_type': sample_type, 
            'metric_type': 'RMSE',
            'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results

def run_OLS_with_clusters(train, validate, target, model_number, model_info, model_results):
    
    # identify the cluster features
    cluster_features = [col for col in train.columns if 'cluster_' in col and 'enc_' not in col]

    for cluster_feature in cluster_features:

        # including the most highly correlated features from exploration
        features = ['scaled_sqft',
                    'scaled_bedroomcnt',
                    'scaled_bathroomcnt',
                    'scaled_fullbathcnt',
                    'scaled_age',
                    'scaled_assessmentyear',
                    'scaled_yearbuilt',
                    'scaled_garagecarcnt',
                    'scaled_garagetotalsqft']

        # adding encoded cluster feature columns to feature set
        for encoded_cluster_column in [col for col in train.columns if f'enc_{cluster_feature}_' in col]:
            features.append(encoded_cluster_column)

        # establish model number
        model_number += 1

        # establish model type
        model_type = 'OLS regression'

        # create a dictionary containing the features and hyperparamters used in this model instance
        dct = {'model_number': model_number,
               'model_type': model_type,
               'features': features,
               'cluster': cluster_feature[8:]}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create the model object and fit to the training sample
        linreg = LinearRegression(normalize=True).fit(x_train, y_train)

        # make predictions for the training sample
        y_pred = linreg.predict(x_train)
        sample_type = 'train'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)

        # make predictions for the validate sample
        y_pred = linreg.predict(x_validate)
        sample_type = 'validate'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results

def run_PolyReg_with_clusters(train, validate, target, model_number, model_info, model_results):
    
    cluster_features = [col for col in train.columns if 'cluster_' in col and 'enc_' not in col]
    
    for degree in range(2, 5):

        for cluster_feature in cluster_features:


            # including the most highly correlated features from exploration
            features = ['scaled_sqft',
                        'scaled_bedroomcnt',
                        'scaled_bathroomcnt',
                        'scaled_fullbathcnt',
                        'scaled_age',
                        'scaled_assessmentyear',
                        'scaled_yearbuilt',
                        'scaled_garagecarcnt',
                        'scaled_garagetotalsqft']

            # adding encoded cluster feature columns to feature set
            for encoded_cluster_column in [col for col in train.columns if f'enc_{cluster_feature}_' in col]:
                features.append(encoded_cluster_column)

            # establish model number
            model_number += 1

            # establish model type
            model_type = 'polynomial regression'

            # create a dictionary containing the features and hyperparamters used in this model instance
            dct = {'model_number': model_number,
                   'model_type': model_type,
                   'features': features,
                   'cluster': cluster_feature[8:],
                   'degree': degree}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            #split the samples into x and y
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]

            # create a polynomial features object
            pf = PolynomialFeatures(degree=degree)

            # fit and transform the data
            x_train_poly = pf.fit_transform(x_train)
            x_validate_poly = pf.fit_transform(x_validate)

            # create the model object and fit to the training sample
            linreg = LinearRegression().fit(x_train_poly, y_train)

            # make predictions for the training sample
            y_pred = linreg.predict(x_train_poly)
            sample_type = 'train'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'RMSE',
                   'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)

            # make predictions for the validate sample
            y_pred = linreg.predict(x_validate_poly)
            sample_type = 'validate'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'RMSE',
                   'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)
            
    return model_number, model_info, model_results

def test_model_8(train, test, target):
    
    # identify features used in the model
    features = ['scaled_sqft',
                'scaled_bedroomcnt',
                'scaled_bathroomcnt',
                'scaled_fullbathcnt',
                'scaled_age',
                'scaled_assessmentyear',
                'scaled_yearbuilt',
                'scaled_garagecarcnt',
                'scaled_garagetotalsqft',
                'enc_cluster_BedBathTaxvaluepersqft_1',
                'enc_cluster_BedBathTaxvaluepersqft_2']
    
    #split the samples into x and y
    x_train = train[features]
    y_train = train[target]

    x_test = test[features]
    y_test = test[target]

    # create the model object and fit to the training sample
    linreg = LinearRegression(normalize=True).fit(x_train, y_train)

    # make predictions for the test sample
    y_pred = linreg.predict(x_test)
    
    print(f'Model #8 RMSE on test sample: {round(sqrt(sk.metrics.mean_squared_error(y_test, y_pred)),7)}')