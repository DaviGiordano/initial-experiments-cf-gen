import umap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap_grid(df, hue_data, n_neighbors_values, min_dist_values):
    # If hue data is not from df, make sure the order of the datapoints is the same.
    num_groups = hue_data.nunique()
    # Set up the figure and axes
    fig, axes = plt.subplots(len(n_neighbors_values), len(min_dist_values), figsize=(20, 20))
    fig.suptitle(f'UMAP Embeddings', fontsize=16)

    # Loop over hyperparameters to create each UMAP plot
    for i, n_neighbors in enumerate(n_neighbors_values):
        for j, min_dist in enumerate(min_dist_values):
            # Initialize and fit UMAP
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            embedding = reducer.fit_transform(df)
            
            # Select the current subplot axis
            ax = axes[i, j]
            
            # Plot UMAP embedding on the axis
            sns.scatterplot(
                x=embedding[:, 0], 
                y=embedding[:, 1], 
                hue=hue_data, 
                palette=sns.color_palette("colorblind")[:num_groups], 
                alpha=0.3, 
                ax=ax, 
                edgecolor=None
            )
            
            # Set plot title and remove axis labels for a cleaner look
            ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
            ax.set_xlabel("UMAP Component 1")
            ax.set_ylabel("UMAP Component 2")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_single_umap(df, hue_data, n_neighbors, min_dist):
    num_groups = hue_data.nunique()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(df)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=hue_data, 
        palette=sns.color_palette("colorblind")[:num_groups], 
        alpha=0.3, 
        edgecolor=None
    )
    
    plt.title(f'UMAP Embedding: n_neighbors={n_neighbors}, min_dist={min_dist}')
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.show()