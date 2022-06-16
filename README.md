## About the Project

### Background

I am a junior data scientists on the Zillow data science team. I through my exploration of or data I have found some features that could be driving the errors in the Zestimates for single unit/single family homes in 2017?". 


Acknowledgement:The dataset was provided by Codeup from the MySequel Database

### Goals

My goal for this project is to create a model that will find what is driving the errors in the Zestimates of single unit properties in 2017 by including clustering methodologies. I will deliver the following in a github repository:

- A clearly named final notebook. This notebook will be what I present and will contain plenty of markdown documentation and cleaned up code.
- A README that explains what the project is, how to reproduce you work, and your notes from project planning
- A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.



| Target   | Definition                                                                           |
| -------- | ------------------------------------------------------------------------------------ |
| logerror | the log of the difference between what the zestimate is and the actual selling price |
------------------------------------


# Initial Hypothesis & Thoughts

## Thoughts

- We could add a new feature?
- Should I turn the continuous variables into booleans?

## Hypothesis

### Hypothesis 1:

    - H0: The mean logerror is the same across all counties
    - Ha: The mean logerror is not the same across all counties

### Hypothesis 2:

    - H0: Log errors for low cost per sqft houses are the same as the log errors for the rest of the houses
    - Ha: Log errors for low cost per sqft houses are different than the log errors for the rest of the houses

### Hypothesis 3:

    - H0: The mean logerror is the same across all bedrooms
    - Ha: The mean logerror is not the same across all bedrooms

### Hypothesis 4:

    - H0: The mean logerror is the same across all heating systems
    - Ha: The mean logerror is the same across all heating systems

### Hypothesis 5:

    - H0: The mean logerror is the same across all size clusters
    - Ha: The mean logerror is not the same across all size clusters

### Hypothesis 6:

    - H0: The mean logerror is the same across all feature clusters
    - Ha: The mean logerror is the same across all feature clusters

### Hypothesis 7:

    - H0: The mean logerror is the same across all value clusters
    - Ha: The mean logerror is the same across all value clusters


# Project Plan: Breaking it Down

- acquire

    - acquire data from MySQL
        - join tables to include transaction date
    - save as a csv and turn into a pandas dataframe
    - summarize the data
    - plot distribution

- prepare

    - address missing data
    - create features
    - split into train, validate, test
    
- explore

    - test each hypothesis
    - plot the continuous variables
    - plot correlation matrix of all variables
    - create clusters and document its usefulness/helpfulness

- model and evaluation

    - which features are most influential: use rfe
    - try different algorithms: LinearRegression, LassoLars, Polynomial Regression
    - evaluate on train
    - evaluate on validate
    - select best model
    - create a model.py that pulls all the parts together.
    - run model on test to verify.

- conclusion

    - summarize findings
    - make recommendations
    - next steps


# How to Reproduce

1. Download data from zillow database in MySQL with Codeup credentials.
2. Install acquire.py, prepare.py and model.py into your working directory.
3. Run a jupyter notebook importing the necessary libraries and functions.
4. Follow along in final_report.ipynb or forge your own exploratory path.
