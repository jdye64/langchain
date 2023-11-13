from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env
from typing import List, Dict, Optional
import httpx
import json
import numpy as np


class NeMoLLMEmbeddings(BaseModel, Embeddings):
    """NeMo LLM embedding models.

    Example:
        .. code-block:: python

            from langchain.embeddings import NeMoLLMEmbeddings
            embedding_model = NeMoEmbeddings(nemo_llm_api_key="my-api-key")

    Example:
        .. code-block:: python
            from langchain.embeddings.openai import OpenAIEmbeddings
            embedding_model = NeMoEmbeddings(
                nemo_llm_api_key="my-api-key",
                nemo_llm_org_id="my-org-id",
                model="e5-large-unsupervised",
            )
            text = "Hello world!"
            embeddings = embedding_model.embed_query(text)

            texts = ["Hello world!", "another string", "embeddings"]
            embeddings = embedding_model.embed_documents(texts)

    """
    nemo_llm_api_key: Optional[str] = None
    nemo_llm_api_host: Optional[str] = None
    nemo_llm_org_id: Optional[str] = None
    model: str = "e5-large-unsupervised"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        # API Key should be readable from environment variable or supplied as parameter.
        nemo_llm_api_key = get_from_dict_or_env(
            values, "nemo_llm_api_key", "NEMO_LLM_API_KEY"
        )
        values["nemo_llm_api_key"] = nemo_llm_api_key

        # API HOST should be readable from environment variable or supplied as parameter.
        # Default to fall back on.
        nemo_llm_api_host = get_from_dict_or_env(
            values,
            "nemo_llm_api_host",
            "NEMO_LLM_API_HOST",
            default="https://api.llm.ngc.nvidia.com/v1",
        )
        values["nemo_llm_api_host"] = nemo_llm_api_host

        # Org Id should be readable from environment variable or supplied as parameter.
        # If None found, this is OK, unless the API Key is associated with
        # more than one LLM organization or
        # more than one LLM team within an LLM organization
        try:
            nemo_llm_org_id = get_from_dict_or_env(
                values, "nemo_llm_org_id", "NEMO_LLM_ORG_ID"
            )
        except ValueError:
            nemo_llm_org_id = None
        values["nemo_llm_org_id"] = nemo_llm_org_id
        return values

    def _get_request_url(self) -> str:
        url = f"{self.nemo_llm_api_host}/embeddings/{self.model}"
        return url

    def _get_request_headers(self) -> Dict[str, str]:
        # nemo_llm_org_id is only required if user is part of
        # more than one LLM organization or
        # more than one LLM team within an LLM organization
        headers = {
            "Authorization": f"Bearer {self.nemo_llm_api_key}",
        }
        if self.nemo_llm_org_id is not None:
            headers["Organization-ID"] = self.nemo_llm_org_id
        return headers

    def _get_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(60.0, connect=10.0)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=self._get_timeout()) as client:
            response = client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json={"content": texts},
            )

        response.raise_for_status()
        return response.json()["embeddings"]
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to NeMo LLM INFORM's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._get_embeddings(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to NeMo LLM INFORM's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """Call out to  NeMo LLM INFORM's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to NeMo LLM INFORM's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        raise NotImplementedError
