import json
import http.client
import logging
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


class NestleEmbeddings:
    """Custom embeddings implementation for Nestle's internal OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client_id: str = "",
        client_secret: str = "",
        api_version: str = "2024-02-01",
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the Nestle Embeddings model.

        Args:
            model: Model name (default: text-embedding-3-small)
            client_id: API client ID
            client_secret: API client secret
            api_version: API version (default: 2024-02-01)
            encoding_format: Format for embeddings (float or base64)
            dimensions: Optional dimension reduction (for text-embedding-3-* models)
        """
        self.model = model
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_version = api_version
        self.encoding_format = encoding_format
        self.dimensions = dimensions

        # API configuration based on the provided URL structure
        self.host = "int-eur-sdr-int-prv.nestle.com"
        self.base_path = (
            "/nlx-eudv-exp-accelerator-openai-api-v1/api/openai/deployments"
        )

    def _make_api_call(self, texts: List[str]) -> List[List[float]]:
        """
        Make API call to get embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Prepare payload
        payload_dict = {"input": texts, "encoding_format": self.encoding_format}

        # Add dimensions if specified (only for text-embedding-3-* models)
        if self.dimensions is not None:
            payload_dict["dimensions"] = self.dimensions

        payload = json.dumps(payload_dict)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        api_path = (
            f"{self.base_path}/{self.model}/embeddings"
            f"?api-version={self.api_version}"
        )

        try:
            conn = http.client.HTTPSConnection(self.host)
            conn.request("POST", api_path, payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            conn.close()

            if res.status != 200:
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"API error {res.status}: {error_msg}")

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except Exception as e:
            logger.error(f"Nestle Embeddings API error: {str(e)}")
            raise

    def embed_documents(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text documents to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

            try:
                embeddings = self._make_api_call(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self._make_api_call([text])
        return embeddings[0]

    def __call__(self, text: str) -> List[float]:
        """
        Allow the class to be called directly for single text embedding.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_query(text)


# Wrapper class for LangChain compatibility
class NestleEmbeddingsLangChain:
    """
    LangChain-compatible wrapper for NestleEmbeddings.
    Implements the interface expected by LangChain's embedding classes.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client_id: str = "",
        client_secret: str = "",
        api_version: str = "2024-02-01",
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the LangChain-compatible embeddings.

        Args:
            model: Model name
            client_id: API client ID
            client_secret: API client secret
            api_version: API version
            dimensions: Optional dimension reduction
        """
        self.embeddings = NestleEmbeddings(
            model=model,
            client_id=client_id,
            client_secret=client_secret,
            api_version=api_version,
            dimensions=dimensions,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embeddings.embed_query(text)


# Example usage
if __name__ == "__main__":
    import os

    # Initialize embeddings
    embeddings = NestleEmbeddings(
        model="text-embedding-3-small",
        client_id=os.getenv("NESTLE_CLIENT_ID"),
        client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
    )

    # Example 1: Embed single query
    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")

    # Example 2: Embed multiple documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
    ]
    doc_embeddings = embeddings.embed_documents(documents)
    print(f"Embedded {len(doc_embeddings)} documents")

    # Example 3: Use with LangChain
    langchain_embeddings = NestleEmbeddingsLangChain(
        client_id=os.getenv("NESTLE_CLIENT_ID"),
        client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
    )

    embeddings_lc = langchain_embeddings.embed_documents(documents)
    print(f"LangChain compatible embeddings: {len(embeddings_lc)}")
