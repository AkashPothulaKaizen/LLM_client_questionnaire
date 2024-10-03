import boto3
import pandas as pd
import io
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import botocore
from typing import List


class DataRetriever:
    def __init__(self, config):
        self.s3_client = boto3.client("s3")
        self.bucket = config.vector_store_bucket
        self.key = config.vector_store_key
        self.model_id = config.embed_model_id
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.n_neighbors = config.n_neighbors

    def embed_content(self, prompt_data: str) -> str:
        """
        Generate embeddings for the given prompt data.
        Parameters:
        prompt_data (str): The input text to be embedded. 
        Returns:
        list: The embedding vector.
        """
        body = json.dumps({"inputText": prompt_data})
        try:
            response = self.bedrock_runtime.invoke_model(
                body=body, modelId=self.model_id
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding")
            return embedding
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == 'AccessDeniedException':
                print(f"\x1b[41m{error.response['Error']['Message']}\n"
                      "To troubleshoot this issue, please refer to the following resources:\n"
                      "https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\n"
                      "https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
            else:
                raise error

    def load_data_from_s3(self) -> None:
        """Load the csv file from S3 bucket (vector_store_bucket)."""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
        file_content = response['Body'].read().decode('utf-8')
        self.df = pd.read_csv(io.StringIO(file_content))

    def get_document_embeddings(self) -> List[List[float]]:
        """Convert the embeddings from string to list."""
        return self.df["embeddings"].apply(eval).to_list()

    def get_query_embeddings(self, query: str) -> List[float]:
        """convert query into embeddings vector."""
        return self.embed_content(query)

    def find_nearest_neighbors(self,
                               document_embeddings: List[List[float]],
                               query_embeddings: List[float]) -> np.ndarray:
        """Find the nearest neighbors for the query embeddings."""
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(np.array(document_embeddings))
        distances, indices = knn.kneighbors(
            np.array(query_embeddings).reshape(1, -1)
        )
        return indices

    def retrieve_data(self, query: str) -> str:
        """Retrieve the most relevant data for the given query."""
        self.load_data_from_s3()
        document_embeddings = self.get_document_embeddings()
        query_embeddings = self.get_query_embeddings(query)
        indices = self.find_nearest_neighbors(
            document_embeddings, query_embeddings
        )
        retrieved_data = []
        for i, idx in enumerate(indices[0]):
            retrieved_data.append(self.df["question_answer_data"].iloc[idx])
        formatted_data = "\n\n".join(retrieved_data)
        return formatted_data
