from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus.exceptions import SchemaNotReadyException


class MilvusClient:
    """
    Handles all Milvus-related operations:
    - Connecting to Milvus
    - Creating collections
    - Inserting data
    - Searching for similar vectors
    - Creating indexes
    """

    def __init__(self, host="localhost", port="19530", collection_name="products"):
        self.collection_name = collection_name

        # Connect to the Milvus server using the given host and port
        connections.connect(alias="default", host=host, port=port)

        try:
            # Try to reference the collection by name
            self.collection = Collection(self.collection_name)

            try:
                # Check if the collection has an index; if not, create it
                if not self.collection.has_index():
                    print(f"⚠ Index not found for collection '{self.collection_name}'. Creating index automatically...")
                    self.create_index()
                # Attempt to load the collection into memory
                self.collection.load()
            except Exception as e:
                # Handle the case where the collection exists but has no index yet
                if "index not found" in str(e):
                    print(f"⚠ Index not found for collection '{self.collection_name}'. You must call create_index() before loading.")
                else:
                    # Raise any unexpected exception
                    raise e
        except SchemaNotReadyException:
            # If the collection does not exist, warn the user and delay collection creation
            print(f"⚠ Collection '{self.collection_name}' not found. Please create it first using create_collection().")
            self.collection = None
    def create_collection(self, dim=384):
        """
        Creates a new collection in Milvus for storing product embeddings.
        If the collection already exists, it is dropped first.
        """
        from pymilvus import list_collections
        if self.collection_name in list_collections():
            Collection(self.collection_name).drop()

        fields = [
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, description="Product Embedding Collection")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"✅ Created Milvus collection: {self.collection_name}")

    def insert_embeddings(self, product_ids, embeddings):
        """
        Inserts a list of embeddings and product IDs into Milvus.

        Args:
            product_ids (List[str]): Product IDs to use as primary keys
            embeddings (List[List[float]]): Corresponding float vector embeddings
        """
        data = [product_ids, embeddings]
        self.collection.insert(data=data)
        print(f"✅ Inserted {len(product_ids)} vectors into Milvus")

    def create_index(self):
        """
        Creates an index on the embedding field to speed up similarity search.
        """
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        print("✅ Index created on embedding field")

    def search_similar(self, vector, top_k=5):
        """
        Searches for similar vectors to the given one.

        Args:
            vector (List[float]): The query vector
            top_k (int): Number of similar results to return

        Returns:
            List[Tuple[str, float]]: List of (product_id, similarity score)
        """
        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["product_id"]
        )
        return [(hit.entity.get("product_id"), hit.distance) for hit in results[0]]