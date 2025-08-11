import matplotlib.pyplot as plt

def plot_silhouette_scores(score_dict):
    """
    Plot Silhouette Scores for different number of clusters.

    Parameters:
    - score_dict (dict): A dictionary where keys are number of clusters (int) and values are silhouette scores (float)
    """
    if not score_dict:
        raise ValueError("score_dict is empty. Provide valid silhouette scores to plot.")

    sorted_scores = sorted(score_dict.items())
    ks, scores = zip(*sorted_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o')
    plt.title("Silhouette Scores by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()