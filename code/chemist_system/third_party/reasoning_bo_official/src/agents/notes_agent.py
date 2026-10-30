#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-05-13 21:17:05
@File: src/agents/notes_agent.py
@IDE: vscode
"""

from typing import Optional, List, Dict, Any, Type, Union
from pydantic import BaseModel, Field
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from src.agents.kg_agent import KGAgent
from src.agents.milvus_agent import MilvusAgent
import json
from camel.messages import BaseMessage
import re


class BaseNotesResponse(BaseModel):
    """Base response format for notes extraction. All integrated classes must return str or List[str]"""

    notes: List[str]


class ReasoningNotesResponse(BaseModel):
    """Structured notes extracted from reasoning data"""

    notes: List[str] = Field(
        ...,
        description="General bullet-point notes extracted from reasoning data. May include summaries of findings, relationships, or principles.",
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Important factual discoveries or observations drawn from the reasoning process.",
    )

    parameter_relationships: List[str] = Field(
        default_factory=list,
        description="Descriptive cause-effect or correlation relationships between different parameters or variables.",
    )

    optimization_principles: List[str] = Field(
        default_factory=list,
        description="Actionable principles or rules that can help optimize the experiment. Should be verifiable and clearly beneficial.",
    )


class CompassNotesResponse(BaseModel):
    """
    General-purpose response format for structured scientific notes.
    Suitable for notes extracted from any experimental description or setup.
    """

    notes: List[str] = Field(
        ...,
        description="Free-form bullet-point notes. General scientific facts, summaries, or highlights.",
    )

    theoretical_background: List[str] = Field(
        default_factory=list,
        description="Scientific principles, mechanisms, or theories underlying the experiment.",
    )

    variable_properties: List[str] = Field(
        default_factory=list,
        description="Descriptions of individual variables’ attributes, roles, or behaviors.",
    )

    variable_relationships: List[str] = Field(
        default_factory=list,
        description="Cause-effect or correlated relationships between different variables or parameters.",
    )


class NotesAgent:
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        kg_agent: Optional[KGAgent] = None,
        milvus_agent: Optional[MilvusAgent] = None,
        system_message: str = "You are an expert at extracting key notes from scientific data or complex reasoning data..",
        tools: Optional[List[Any]] = None,
    ):
        """
        Initialize Notes Agent for processing reasoning data

        Args:
            model: Backend model for processing
            kg_agent: Pre-initialized KGAgent instance
            milvus_agent: Pre-initialized MilvusAgent instance
            system_message: System message for the chat agent
            tools: List of tools for the agent to use
        """
        self.chat_agent = ChatAgent(
            model=model, system_message=system_message, tools=tools or []
        )
        self.kg_agent = kg_agent or KGAgent(model=model)
        self.milvus_agent = milvus_agent or MilvusAgent()

        self._init_prompt_templates()

    def _init_prompt_templates(self):
        """Initialize default prompt templates"""
        self.DEFAULT_REASONING_NOTES_PROMPT = """
        Analyze the following reasoning data and extract structured scientific notes.

        Guidelines:
        - Each item should be a clear and self-contained bullet point.
        - Only use information present in the reasoning. Do not speculate.
        - Keep the content factual, concise, and helpful.

        Focus on the following aspects:
        1. **Key findings** — Verifiable results, conclusions, or observations grounded in reasoning.
        2. **Parameter relationships** — How different variables or conditions affect each other (cause-effect, correlation).
        3. **Optimization principles** — Rules or strategies suggested by the reasoning that could improve experimental results.
        4. **General notes** — Summarized insights that don't fall into the above categories.

        Reasoning data:
        {input}
        """

        self.DEFAULT_COMPASS_NOTES_PROMPT = """
        Extract and structure scientific notes from the following experimental setup.

        Guidelines:
        - Use bullet-point style strings in each list.
        - Base your output only on the given input.
        - Do not fabricate information not present in the experiment.

        Focus on the following aspects:
        1. **Theoretical background** — Scientific principles, hypotheses, or mechanisms implied or explicitly stated.
        2. **Variable properties** — Roles, attributes, or behaviors of each variable/component involved.
        3. **Variable relationships** — Any cause-effect, dependency, or correlation observed or described.
        4. **General notes** — Additional relevant scientific information or implicit assumptions.

        Experimental setup:
        {input}
        """

    def extract_reasoning_notes(
        self,
        reasoning_data: str,
        save_schema: Type[BaseNotesResponse] = BaseNotesResponse,
        prompt: Optional[str] = None,
    ) -> BaseNotesResponse:
        """
        Extract key notes from reasoning data and store to storage systems

        Args:
            reasoning_data: Raw reasoning output to process
            response_model: Custom response model class
            prompt: Custom extraction prompt (uses default if None)

        Returns:
            Structured response with extracted notes
        """
        final_prompt = (prompt or self.DEFAULT_REASONING_NOTES_PROMPT).format(
            input=reasoning_data
        )

        response = self.chat_agent.step(
            final_prompt, response_format=save_schema
        )

        self._store_notes(response)
        return response

    def extract_compass_notes(
        self,
        experiment_data: str,
        save_schema: Type[BaseNotesResponse] = BaseNotesResponse,
        prompt: Optional[str] = None,
        enable_research: bool = False,
    ) -> BaseNotesResponse:
        """
        Extract and store experimental setup information

        Args:
            experiment_data: Description of experimental setup
            save_schema: Response schema class
            prompt: Custom extraction prompt (uses default if None)
            enable_research: Whether to enable web research for missing info

        Returns:
            Structured response with extracted notes
        """
        final_prompt = (prompt or self.DEFAULT_COMPASS_NOTES_PROMPT).format(
            input=experiment_data
        )

        # TODO Add web search tool
        if enable_research:
            final_prompt += (
                "\n\nResearch and supplement any missing critical information."
            )

        response = self.chat_agent.step(
            final_prompt,
            response_format=save_schema,
        )
        print(f"response content: {response}")
        print(f"\nresponse type: {type(response)}")
        self._store_notes(response)
        return response

    def _store_notes(self, response: Union[BaseNotesResponse, List[str]]):
        """Internal method to store notes to both knowledge systems

        Args:
            response: Can be one of the following types:
                - camel.ChatAgentResponse: ChatAgentResponse.msg is camel.BaseMessage (contains JSON-formatted content)
                - String list, all fields must be str or List[str]
        """
        if isinstance(response.msg, BaseMessage):
            try:
                # Extract JSON content part (remove possible code block markers)
                content = (
                    response.msg.content.replace('```json\n', '')
                    .replace('```', '')
                    .strip()
                )
                data = json.loads(content)
                # Collect all possible text fields
                text_parts = "\n".join(
                    str(item)
                    for value in data.values()
                    for item in (value if isinstance(value, list) else [value])
                )
            except (json.JSONDecodeError, AttributeError) as e:
                raise ValueError(
                    f"Failed to parse message content: {e}"
                ) from e

        # Handle pure list input
        elif isinstance(response, (list, tuple)):
            text_parts = "\n".join(str(item) for item in response)
        else:
            raise TypeError(
                f"Unsupported type, response attributes must be str or List[str]"
            )

        if text_parts:
            self.kg_agent.parse(text_parts)
            self.milvus_agent.parse(text_parts)

    def query_notes(
        self, query: str, top_k: int = 3, similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query relevant notes from storage systems, including KG and vector DB

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            Combined results from KG and vector DB
        """
        kg_results = self.kg_agent.run_retriever(query)
        milvus_results = self.milvus_agent.run_retriever(
            query, top_k=top_k, similarity_threshold=similarity_threshold
        )

        return {"knowledge_graph": kg_results, "vector_db": milvus_results}

    def structured_query(
        self, prompt: str, response_schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Execute a structured query with custom response format

        Args:
            prompt: User query
            response_model: Custom pydantic model for response

        Returns:
            Structured response matching the provided model
        """
        return self.chat_agent.step(prompt, response_format=response_schema)

    @staticmethod
    def format_retrieved_notes(results: Dict[str, Any]) -> str:
        """Format retrieved notes into a well-structured string with enhanced robustness

        Args:
            results: Dictionary containing retrieval results from knowledge graph and vector DB

        Returns:
            Formatted string with clearly categorized retrieval results
        """
        formatted = []

        # Process knowledge graph results
        if results.get("knowledge_graph"):
            try:
                formatted.append(
                    "=== Retrieved_context from knowledge graph ==="
                )
                kg_relations = set()  # Using set for deduplication

                # Extract and normalize knowledge graph relations
                for relation in results["knowledge_graph"]:
                    if isinstance(relation, str):
                        # Extract key information from the relation
                        match = re.search(
                            r"Node (.+?) \(.*?\) has relationship (.+?) with Node (.+?) \(",
                            relation,
                        )
                        if match:
                            entity1, rel_type, entity2 = match.groups()
                            kg_relations.add(
                                f"{entity1} → {rel_type} → {entity2}"
                            )
                        else:
                            kg_relations.add(
                                relation
                            )  # Keep original format if parsing fails

                # Add sorted relations
                formatted.extend(sorted(kg_relations))

            except Exception as e:
                print(
                    f"Warning: Error processing knowledge graph results - {str(e)}"
                )
                formatted.append("(Error parsing knowledge graph data)")

        # Process vector DB results
        if results.get("vector_db"):
            try:
                formatted.append(
                    "\n=== Retrieved_context from Milvus Database ==="
                )

                for item in results["vector_db"]:
                    if not isinstance(item, dict):
                        continue

                    # Get content from either 'text' or 'content' field
                    content = item.get("text") or item.get("content") or ""
                    if content:
                        formatted.append(f"- {content.strip()}")

            except Exception as e:
                print(
                    f"Warning: Error processing vector DB results - {str(e)}"
                )
                formatted.append("(Error parsing vector database data)")

        return (
            "\n".join(filter(None, formatted))
            if formatted
            else "No relevant results found"
        )
