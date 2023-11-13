from typing import Any, List, Dict, Mapping, Optional, Iterator, AsyncIterator
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
import httpx
import json


class ChatNeMoLLM(BaseChatModel, BaseModel):
    nemo_llm_api_key: Optional[str] = None
    nemo_llm_api_host: Optional[str] = None
    nemo_llm_org_id: Optional[str] = None
    model_name: str = "gpt-43b-905"
    temperature: float = 0.5
    top_k: int = 2
    top_p: float = 1
    random_seed: int = 0
    tokens_to_generate: int = 100
    repetition_penalty: float = 1.0
    beam_search_diversity_rate: float = 0.0
    beam_width: int = 1
    length_penalty: float = 1.0
    customization: Optional[str] = None
    streaming: bool = False

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

        # The remaining parameters are numeric and have ranges of expected values.

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if (
            values["tokens_to_generate"] is not None
            and values["tokens_to_generate"] <= 0
        ):
            raise ValueError("tokens_to_generate must be greater than zero")

        if values["random_seed"] is not None and values["random_seed"] < 0:
            raise ValueError("random_seed must be 0 or positive")

        if (
            values["repetition_penalty"] is not None
            and not 1 <= values["repetition_penalty"] <= 2
        ):
            raise ValueError("top_p must be in the range [1.0, 2.0]")

        if (
            values["beam_search_diversity_rate"] is not None
            and not 0 <= values["beam_search_diversity_rate"] <= 1
        ):
            raise ValueError(
                "beam_search_diversity_rate must be in the range [0.0, 1.0]"
            )

        if (
            values["beam_width"] is not None
            and values["beam_width"] != 1
            and values["beam_width"] != 2
            and values["beam_width"] != 3
            and values["beam_width"] != 4
        ):
            raise ValueError("beam_width must be 1, 2, 3, or 4")

        if (
            values["length_penalty"] is not None
            and not 0 <= values["length_penalty"] <= 1
        ):
            raise ValueError("length_penalty must be in the range [0.0, 1.0]")

        return values

    @property
    def _llm_type(self) -> str:
        return "nemollm-chat"

    def _get_request_url(self) -> str:
        if self.customization is None:
            url = f"{self.nemo_llm_api_host}/models/{self.model_name}/chat"
        else:
            url = f"{self.nemo_llm_api_host}/models/{self.model_name}/customizations/{self.customization}/chat"
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
        if self.streaming:
            headers["x-stream"] = "true"
        return headers

    def _get_request_json(
        self, chat_context: List[Dict[str, str]], stop: Optional[List[str]] = None
    ) -> Dict:
        if stop is None:
            stop = []

        return {
            "chat_context": chat_context,
            "logprobs": False,
            "stop": stop,
            **{
                k: v for (k, v) in self._identifying_params.items() if k != "model_name"
            },
        }

    def _get_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(60.0, connect=10.0)

    def _messages_to_chat_context(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        chat_context = []
        for message in messages:
            ChatMessage
            # convert list of messages to the format expected by NeMo API
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, ChatMessage):
                assert (
                    message.role == "system"
                    or message.role == "assistant"
                    or message.role == "user"
                )
                role = message.role
            else:
                raise ValueError(
                    f"message must be of type SystemMessage, AIMessage, HumanMessage, or ChatMessage. Given type: {type(message)}"
                )
            chat_context.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )
        return chat_context

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if "chat_context" in kwargs:
            chat_context = kwargs["chat_context"]
        else:
            chat_context = self._messages_to_chat_context(messages)

        with httpx.Client(timeout=self._get_timeout()) as client:
            with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(chat_context, stop),
            ) as r:
                for json_line in r.iter_lines():
                    if not json_line:
                        break
                    text = json.loads(json_line)["text"]
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if "chat_context" in kwargs:
            chat_context = kwargs["chat_context"]
        else:
            chat_context = self._messages_to_chat_context(messages)

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            async with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(chat_context, stop),
            ) as r:
                async for json_line in r.aiter_lines():
                    if not json_line:
                        break
                    text = json.loads(json_line)["text"]
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_context = self._messages_to_chat_context(messages)

        if self.streaming:
            completion = ""
            for chunk in self._stream(messages, stop, run_manager, chat_context=chat_context, **kwargs):
                completion += chunk.text
            completion = completion
        else:
            with httpx.Client(timeout=self._get_timeout()) as client:
                response = client.post(
                    url=self._get_request_url(),
                    headers=self._get_request_headers(),
                    json=self._get_request_json(chat_context, stop),
                )

            response.raise_for_status()
            completion = response.json()["text"]

        message = AIMessage(content=completion)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"chat_context": chat_context},
                )
            ],
            llm_output={
                "url": self._get_request_url(),
                "headers": self._get_request_headers(),
                "model_name": self.model_name,
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_context = self._messages_to_chat_context(messages)

        if self.streaming:
            completion = ""
            async for chunk in self._astream(messages, stop, run_manager, chat_context=chat_context, **kwargs):
                completion += chunk.text
            completion = completion
        else:
            async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
                response = await client.post(
                    url=self._get_request_url(),
                    headers=self._get_request_headers(),
                    json=self._get_request_json(chat_context, stop),
                )

            response.raise_for_status()
            completion = response.json()["text"]

        message = AIMessage(content=completion)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"chat_context": chat_context},
                )
            ],
            llm_output={
                "url": self._get_request_url(),
                "headers": self._get_request_headers(),
                "model_name": self.model_name,
            },
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "random_seed": self.random_seed,
            "tokens_to_generate": self.tokens_to_generate,
            "repetition_penalty": self.repetition_penalty,
            "beam_search_diversity_rate": self.beam_search_diversity_rate,
            "beam_width": self.beam_width,
            "length_penalty": self.length_penalty,
        }
