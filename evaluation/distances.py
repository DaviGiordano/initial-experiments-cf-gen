import numpy as np

def gowers_distance(point1, point2, numerical_cols, categorical_cols):
    # Initialize the distance
    distance = 0
    total_weights = 0
    
    # Calculate the distance for numerical columns
    for col in numerical_cols:
        # Normalize the difference by the range of the values for that column
        range_val = np.ptp([point1[col], point2[col]])  # range of the column values
        if range_val == 0:  # Avoid division by zero
            diff = 0
        else:
            diff = abs(point1[col] - point2[col]) / range_val
        distance += diff
        total_weights += 1

    # Calculate the distance for categorical columns
    for col in categorical_cols:
        diff = 0 if point1[col] == point2[col] else 1
        distance += diff
        total_weights += 1
    
    # Calculate the Gower distance
    gower_distance = distance / total_weights
    return gower_distance
