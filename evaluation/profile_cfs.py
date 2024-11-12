import matplotlib.pyplot as plt
import os

def plot_distribution_comparison(factual_df, counterfactual_df, column, sensitive_feature, save_fpath=None):
    """
    Plots the distribution of a continuous feature across factual and counterfactual datasets,
    grouped by a sensitive feature.
    
    Parameters:
    - factual_df: DataFrame containing the factual dataset.
    - counterfactual_df: DataFrame containing the counterfactual dataset.
    - column: The name of the continuous feature column to plot.
    - sensitive_feature: The name of the sensitive feature to group by.
    - save_fpath: Optional file path to save the plot.
    """
    
    # Prepare data and get unique values of the sensitive feature
    factual_grouped = factual_df[[column, sensitive_feature]].dropna()
    counterfactual_grouped = counterfactual_df[[column, sensitive_feature]].dropna()
    unique_sensitive_values = factual_grouped[sensitive_feature].unique()
    
    # Set up the plot
    fig = plt.figure()
    male_fact, female_fact = fig.subplots(1, 2)
    
    for value in unique_sensitive_values:
        # Plot for factual data
        factual_subset = factual_grouped[factual_grouped[sensitive_feature] == value]
        factual_subset[column].plot(kind='kde', label=f'Factual {sensitive_feature}={value}', linestyle='-', alpha=0.5)
        
        # Plot for counterfactual data
        counterfactual_subset = counterfactual_grouped[counterfactual_grouped[sensitive_feature] == value]
        counterfactual_subset[column].plot(kind='kde', label=f'Counterfactual {sensitive_feature}={value}', linestyle='--', alpha=0.5)
    
    # Customize the plot
    plt.title(f'Comparison of {column} Distributions (Factual vs Counterfactual)')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    
    # Save or display plot
    if save_fpath:
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
        plt.savefig(save_fpath, bbox_inches='tight')
        print(f"Plot saved to {save_fpath}")
    else:
        plt.show()
    plt.close()
