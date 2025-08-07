from manager.embedding_manager import EmbeddingManager
from embedding.customer_embedder import CustomerEmbedder
from clustering.customer_clustering_by_embeddings import CustomerClustering


def main():
    # Initialize the embedding manager
    manager = EmbeddingManager(product_csv_path="../data/enriched_retail.csv")
    # Step 1: Generate embeddings and save to pickle
    manager.generate_product_embeddings()

    # Step 2: Create a new Milvus collection (if not already created)
    manager.create_milvus_collection()

    # Step 3: Insert embeddings into Milvus
    manager.insert_embeddings_to_milvus()

    # Step 4: Create an index for fast similarity search
    manager.create_index()

    # Step 5: Generate customer embeddings
    customer_embedder = CustomerEmbedder()
    customer_embedder.generate_customer_embeddings(weighting_strategy="time_price_quantity", time_decay_lambda=0.001)

    # Step 6: Cluster customers using a unified pipeline
    clusterer = CustomerClustering()
    clusterer.load_embeddings()
    clusterer.run_clustering_pipeline(
        method="filtered_kmeans",     # Options: "kmeans", "dbscan", "filtered_kmeans"
        dbscan_eps=0.5,
        dbscan_min_samples=5,
        kmeans_n_clusters=3
    )
    clusterer.save_results()

    print("✅ Embedding pipeline completed.")

if __name__ == "__main__":
    main()
