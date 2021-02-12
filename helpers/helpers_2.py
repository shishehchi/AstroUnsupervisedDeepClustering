def get_distance_map(desom, X):

    #Get Predicted Labels
    y_pred = desom.predict(X)

    #Distance map
    # i - point in X(data)
    # j - Assigned CELL on SOM
    distance_map = desom.map_dist(y_pred)
    
    return distance_map