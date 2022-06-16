import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
random_state = 42

def target_correlations(train, target):
    '''
    This function takes in the zillow train sample and uses pandas and seaborn to create a
    ordered list and heatmap of the correlations between the various quantitative feeatures and the target. 
    '''
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(train.corr().abs()[target]).sort_values(by=target, ascending=False)
    # rename the correlation column
    corr.columns = ['correlation (abs)']
    # establish figure size
    plt.figure(figsize=(10,8))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True)
    # establish a plot title
    plt.title('Features\' Correlation with Target')
    # display the plot
    plt.show()

def viz_cluster_BedBath(train):

    # visualize the clusters
    sns.relplot(data=train, x='bedroomcnt', y='bathroomcnt', hue='cluster_BedBath', palette='bright')
    plt.show()

def viz_logerror_BedBath(train):

    # visualize logerrror by bed_bath_cluster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plt.title('logerror by BedBath cluster')
    sns.histplot(train[train.cluster_BedBath == 'mid'].logerror, color='blue', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBath == 'low'].logerror, color='orange', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBath == 'high'].logerror, color='green', alpha = .5, ax=ax1)
    sns.boxplot(data=train, x='cluster_BedBath', y='logerror', ax=ax2)
    plt.show()

def test_cluster_BedBath(train):

    # testing whether there is a significant difference in logerror among the clusters
    H0 = 'There is no significant difference in logerror among the BedBath clusters.'
    H1 = 'There is a significant difference in logerror among the BedBath clusters.'

    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_BedBath == 'mid'].logerror, 
                             train[train.cluster_BedBath == 'low'].logerror, 
                             train[train.cluster_BedBath == 'high'].logerror)
    print()
    print('H0: ', H0)
    print('H1: ', H1)
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0')
    else: 
        print('Fail to reject H0')

def viz_cluster_BedBathSqft(train):
    # visualize the clusters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=train, x='bedroomcnt', y='bathroomcnt', hue='cluster_BedBathSqft', palette='bright', ax=ax1)
    sns.scatterplot(data=train, x='bedroomcnt', y='sqft', hue='cluster_BedBathSqft', palette='bright', ax=ax2)
    sns.scatterplot(data=train, x='bathroomcnt', y='sqft', hue='cluster_BedBathSqft', palette='bright', ax=ax3)
    plt.show()

def viz_logerror_BedBathSqft(train):
    # visualize logerror by bed_bath_sqft cluster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plt.title('logerror by BedBathSqft cluster')
    sns.histplot(train[train.cluster_BedBathSqft == 'mid'].logerror, color='blue', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBathSqft == 'low'].logerror, color='orange', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBathSqft == 'high'].logerror, color='green', alpha = .5, ax=ax1)
    sns.boxplot(data=train, x='cluster_BedBathSqft', y='logerror', ax=ax2)
    plt.show()

def test_cluster_BedBathSqft(train):
    # testing whether there is a significant difference in logerror among the clusters
    H0 = 'There is no significant difference in logerror among the BedBathSqft clusters.'
    H1 = 'There is a significant difference in logerror among the BedBathSqft clusters.'

    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_BedBathSqft == 'mid'].logerror,
                             train[train.cluster_BedBathSqft == 'low'].logerror,
                             train[train.cluster_BedBathSqft == 'high'].logerror)
    print()
    print('H0: ', H0)
    print('H1: ', H1)
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0')
    else: 
        print('Fail to reject H0')

def viz_multiple_clusters_LatLong(train):

    fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True, sharex=True)

    for i, k in enumerate(range(3,9)):

        # creating clusters with KMeans
        x = train[['scaled_latitude', 'scaled_longitude']]
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(x)
        train['cluster'] = kmeans.predict(x)

        # visualize the clusters
        y = int(i / 3)
        x = i % 3
        ax = sns.scatterplot(data=train, x='longitude', y='latitude', hue='cluster',
                             palette='bright', ax=axes[y,x])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def viz_logerror_LatLong(train):
    # creating visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alpha = .3 #transparency for histplot
    sns.scatterplot(data=train, x='longitude', y='latitude', hue='cluster_LatLong', palette='bright', ax=axes[0])
    sns.histplot(train[train.cluster_LatLong == 'west'].logerror, color='orange', alpha=alpha, ax=axes[1])
    sns.histplot(train[train.cluster_LatLong == 'north'].logerror, color='blue', alpha=alpha, ax=axes[1])
    sns.histplot(train[train.cluster_LatLong == 'central'].logerror, color='green', alpha=alpha, ax=axes[1])
    sns.histplot(train[train.cluster_LatLong == 'east'].logerror, color='red', alpha=alpha, ax=axes[1])
    sns.boxplot(data=train, x='cluster_LatLong', y='logerror', ax=axes[2])
    plt.show()

def viz_cluster_BedBathTaxvaluepersqft(train):
    # visualize the clusters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=train, x='bedroomcnt', y='bathroomcnt', hue='cluster_BedBathTaxvaluepersqft', palette='bright', ax=ax1)
    sns.scatterplot(data=train, x='bedroomcnt', y='taxvalue_per_sqft', hue='cluster_BedBathTaxvaluepersqft', palette='bright', ax=ax2)
    sns.scatterplot(data=train, x='bathroomcnt', y='taxvalue_per_sqft', hue='cluster_BedBathTaxvaluepersqft', palette='bright', ax=ax3)
    plt.show()

def viz_logerror_BedBathTaxvaluepersqft(train):
    # visualize logerror by bed_bath_sqft cluster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plt.title('logerror by BedBathTaxvaluepersqft cluster')
    sns.histplot(train[train.cluster_BedBathTaxvaluepersqft == '2'].logerror, color='blue', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBathTaxvaluepersqft == '0'].logerror, color='orange', alpha = .5, ax=ax1)
    sns.histplot(train[train.cluster_BedBathTaxvaluepersqft == '1'].logerror, color='green', alpha = .5, ax=ax1)
    sns.boxplot(data=train, x='cluster_BedBathTaxvaluepersqft', y='logerror', ax=ax2)
    plt.show()

def test_cluster_BedBathTaxvaluepersqft(train):
    # testing whether there is a significant difference in logerror among the clusters
    H0 = 'There is no significant difference in logerror among the BedBathTaxvaluepersqft clusters.'
    H1 = 'There is a significant difference in logerror among the BedBathTaxvaluepersqft clusters.'

    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_BedBathTaxvaluepersqft == '0'].logerror,
                             train[train.cluster_BedBathTaxvaluepersqft == '1'].logerror,
                             train[train.cluster_BedBathTaxvaluepersqft == '2'].logerror)
    print()
    print('H0: ', H0)
    print('H1: ', H1)
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0')
    else: 
        print('Fail to reject H0')

def test_cluster_LatLong(train):
    # testing whether there is a significant difference in logerror among the clusters
    H0 = 'There is no significant difference in logerror among the LatLong clusters.'
    H1 = 'There is a significant difference in logerror among the LatLong clusters.'

    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_LatLong == 'north'].logerror,
                             train[train.cluster_LatLong == 'west'].logerror,
                             train[train.cluster_LatLong == 'central'].logerror,
                             train[train.cluster_LatLong == 'east'].logerror)
    print()
    print('H0: ', H0)
    print('H1: ', H1)
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0')
    else: 
        print('Fail to reject H0')