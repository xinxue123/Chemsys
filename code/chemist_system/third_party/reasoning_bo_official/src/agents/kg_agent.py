#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-13 11:19:39
@File: src/agent/kg_agent.py
@IDE: vscode
@Description:
    Knowledge Graph Agent using Neo4j
"""
from camel.storages import Neo4jGraph
from camel.storages.graph_storages import GraphElement
from camel.agents import KnowledgeGraphAgent, ChatAgent
from camel.models import BaseModelBackend
from unstructured.documents.elements import Element
from typing import Optional, Any, Union, List
from camel.loaders import UnstructuredIO
import os
import warnings
from urllib.parse import urlparse

from src.config import Config

config = Config()


class KGAgent:
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
    ):
        """
        Initialize the KGAgent with CAMEL and the specified model backend.

        Args:
            model (Optional[BaseModelBackend]): The model backend.
        """
        self.uio = UnstructuredIO()
        self.camel_kg_agent = KnowledgeGraphAgent(model=model)
        self.neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URL,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
        )

    def pre_parse(
        self,
        content: Union[str, 'Element'],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Parse unstructured content into knowledge graph-related content. Automatically detects
        content type and performs chunking if necessary.

        Args:
            content (Union[str, 'Element']): The input content to parse.
            prompt (Optional[str]): Additional prompt for parsing (if needed).

        Returns:
            str: Parsed content.
        """
        if isinstance(content, Element):
            return self._process_single_element(content)

        elif isinstance(content, str):
            parsed_url = urlparse(content)
            is_url = all([parsed_url.scheme, parsed_url.netloc])

            if is_url or os.path.exists(content):
                return self._process_url_or_file(content, is_url)
            else:
                return self._process_plain_text(content)

        return ""

    def _process_single_element(self, element: Element) -> str:
        """Process a single Element."""
        return self.camel_kg_agent.run(
            element=element,
            parse_graph_elements=False,
        )

    def _process_plain_text(self, text: str) -> str:
        """Process plain text content."""
        print(f"Parsed content as plain text: {text}")
        element = self.uio.create_element_from_text(text=text)
        return self._process_single_element(element)

    def _process_url_or_file(self, path: str, is_url: bool) -> str:
        """Process URL or file content."""
        print(f"Parsed content from {'URL' if is_url else 'file'}: {path}")
        element_lists = self.uio.parse_file_or_url(input_path=path) or []
        if not element_lists:
            warnings.warn(f"No elements were extracted from: {path}")
            return ""

        chunk_elements = self.uio.chunk_elements(
            chunk_type="chunk_by_title", elements=element_lists
        )

        results = []
        for chunk in chunk_elements:
            parsed = self.camel_kg_agent.run(
                element=chunk,
                parse_graph_elements=False,
            )
            results.append(parsed)

        return "\n".join(results)

    def validate(self, content: str) -> str:
        """
        Validate the pre-parsed content for accuracy and remove redundant information.
        Additional strategies such as ChatAgent can be used here.

        Args:
            content (str): The pre-parsed content.

        Returns:
            str: Validated content.
        """
        return content

    def parse(
        self,
        content: str,
    ) -> None:
        """
        Parse the content, convert it into nodes and relationships, and save the knowledge graph
        elements to Neo4j.

        Args:
            content (str): The content to parse.
        """
        pre_parsed = self.pre_parse(content=content)
        validated_content = self.validate(content=pre_parsed)
        elements = self.uio.create_element_from_text(validated_content)
        graph_element = self.camel_kg_agent.run(
            element=elements,
            parse_graph_elements=True,
        )
        self.neo4j_graph.add_graph_elements(
            graph_elements=[graph_element],
        )
        print(f"Save successfully, content: {graph_element}")
        return graph_element

    def run_retriever(
        self,
        query: str,
    ) -> Any:
        """
        Run retrieval and inference on the knowledge graph.

        Args:
            query (str): The query to execute.

        Returns:
            Any: Retrieved results.
        """
        query_element = self.uio.create_element_from_text(
            text=query,
        )
        ans_element = self.camel_kg_agent.run(
            query_element, parse_graph_elements=True
        )
        kg_result = []
        for node in ans_element.nodes:
            n4j_query = f"""
        MATCH (n {{id: '{node.id}'}})-[r]->(m)
        RETURN 'Node ' + n.id + ' (label: ' + labels(n)[0] + ') has relationship ' + type(r) + ' with Node ' + m.id + ' (label: ' + labels(m)[0] + ')' AS Description
        UNION
        MATCH (n)<-[r]-(m {{id: '{node.id}'}})
        RETURN 'Node ' + m.id + ' (label: ' + labels(m)[0] + ') has relationship ' + type(r) + ' with Node ' + n.id + ' (label: ' + labels(n)[0] + ')' AS Description
        """
            result = self.neo4j_graph.query(query=n4j_query)
            kg_result.extend(result)

        kg_result = [item['Description'] for item in kg_result]

        return kg_result

    def update(self, content: Union[str, "Element"], **kwargs) -> None:
        """Update the Knowledge Graph."""
        pass
