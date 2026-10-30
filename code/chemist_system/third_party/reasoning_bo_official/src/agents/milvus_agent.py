#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-12 18:02:22
@File: src/agent/milvus_agent.py
@IDE: vscode
@Description:
    Milvus Agent
"""

from typing import Optional, Dict, Union, List
from camel.storages.vectordb_storages import MilvusStorage
from camel.embeddings import OpenAIEmbedding
from camel.retrievers import VectorRetriever
from camel.storages.vectordb_storages import VectorDBStatus
import logging
from src.config import Config

logger = logging.getLogger(__name__)
config = Config()


class MilvusAgent:
    def __init__(
        self,
        vector_dim: int = 1536,  # OpenAI embedding dimension
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the MilvusAgent.

        Args:
            vector_dim (int): Vector dimension, default is 1536 for OpenAI embeddings.
            collection_name (Optional[str]): Name of the collection in Milvus.
        """
        self.storage = MilvusStorage(
            vector_dim=vector_dim,
            url_and_api_key=(config.MILVUS_URL, ""),
            collection_name=collection_name,
        )

        self.retriever = VectorRetriever(
            embedding_model=OpenAIEmbedding(
                url=config.OPENAI_API_BASE, api_key=config.OPENAI_API_KEY
            ),
            storage=self.storage,
        )

    def parse(
        self,
        content: Union[str, List[str]],
    ) -> None:
        """
        Parse and store content into Milvus. Accepts URLs, strings, or local file paths.

        Args:
            content (Union[str, List[str]]): Content to be parsed and stored.
        """
        self.retriever.process(content=content)

    def run_retriever(
        self, query: str, top_k: int = 1, similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Perform retrieval and return results.

        Args:
            query (str): Query string for retrieval.
            top_k (int): Number of top results to return.
            similarity_threshold (float): Minimum similarity score threshold.

        Returns:
            List[Dict]: Retrieved results.
        """
        retrieved_results = self.retriever.query(
            query=query, top_k=top_k, similarity_threshold=similarity_threshold
        )
        return retrieved_results

    def delete(self, ids: List[str]):
        """
        Delete entries by their IDs.

        Args:
            ids (List[str]): List of IDs to delete.
        """
        self.storage.delete(ids=ids)

    def status(self) -> VectorDBStatus:
        """
        Get the status of the Milvus storage.

        Returns:
            VectorDBStatus: Status of the Milvus storage.
        """
        return self.storage.status()

    def clear(self) -> None:
        """
        Clear all data from the Milvus storage.
        """
        self.storage.clear()
