from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import httpx
import json
import numpy as np

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)



class NemoInformRetriever(BaseRetriever):
    kbid: str
    nemo_llm_api_key: Optional[str] = None
    nemo_llm_api_host: Optional[str] = None
    nemo_llm_org_id: Optional[str] = None

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
        url = f"{self.nemo_llm_api_host}/knowledge_bases/{self.kbid}/query"
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

    def _retrieve_content(self, query: str) -> Dict:
        with httpx.Client(timeout=self._get_timeout()) as client:
            response = client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json={"query": query, "top_k":3},
            )

        response.raise_for_status()
        return response.json()

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        content = self._retrieve_content(query)
        docs = []
        for c in content["content"]:
            doc = Document(page_content=c["text"])
            docs.append(doc)
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("not implemented.")

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        raise NotImplementedError("not implemented.")

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        raise NotImplementedError("not implemented.")
