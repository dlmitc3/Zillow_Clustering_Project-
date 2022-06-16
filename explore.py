import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def elbow_plot(cluster_vars, X_train_scaled):
    # elbow method to identify good k for us
    ks = range(2,20)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()


def run_kmeans(k, cluster_vars, cluster_col_name, X_train_scaled):
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train_scaled.index)
    
    return train_clusters, kmeans


def add_to_train(train_clusters, cluster_col_name, X_train, X_train_scaled):
    # concatenate cluster id
    X_train = pd.concat([X_train, train_clusters], axis=1)
    
    # concatenate cluster id
    X_train_scaled = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)
                               
    return X_train, X_train_scaled