import os

import matplotlib.pyplot as plt

def plot_categorical_distribution(df, column, sensitive_feature, save_fpath=None):
    df_grouped = df.groupby([column, sensitive_feature]).size().unstack().fillna(0)
    df_grouped.plot(kind='barh', stacked=False, title=f'Distribution of {column} by {sensitive_feature}')
    plt.xlabel('Frequency')
    plt.ylabel(column)
            
    if save_fpath:
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
        plt.savefig(save_fpath, bbox_inches='tight')
        print(f"Plot saved to {save_fpath}")
    else:
        plt.show()
    plt.close()

def plot_continuous_distribution(df, column, sensitive_feature, save_fpath=None):
    df_grouped = df[[column, sensitive_feature]].dropna()
    unique_sensitive_values = df_grouped[sensitive_feature].unique()
    
    for value in unique_sensitive_values:
        subset = df_grouped[df_grouped[sensitive_feature] == value]
        subset[column].plot(kind='kde', label=f'{sensitive_feature}={value}', alpha=0.5)
    
    plt.title(f'Distribution of {column} by {sensitive_feature}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    
    if save_fpath:
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
        plt.savefig(save_fpath, bbox_inches='tight')
        print(f"Plot saved to {save_fpath}")
    else:
        plt.show()
    plt.close()

