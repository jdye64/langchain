from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, Optional
from langchain.schema import LLMResult
import httpx

CONTEXT_WINDOWS = {
    "gpt5b": 2048,
    "gpt20b": 2048,
    "gpt-43b-001": 4096,
    "gpt-43b-002": 4096,
    "gpt-43b-903": 4096,
    "gpt530b": 2048,
}


class NeMoLLMCallbackHandler(BaseCallbackHandler):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    context_window: int = 0

    def _get_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(60.0, connect=10.0)

    def _count_tokens(
        self,
        url: str,
        headers: Dict[str, str],
        model_name: str,
        prompt: Optional[str],
        chat_context: Optional[Dict[str, str]],
        completion: str,
    ):
        if self.context_window == 0 and model_name in CONTEXT_WINDOWS:
            self.context_window = CONTEXT_WINDOWS[model_name]

        # the API endpoint for count_tokens depends on the URL used to make the completions/chat request
        if url.endswith("/completions"):
            count_tokens_url = url[: -len("/completions")] + "/count_tokens"
            json_body_prompt = {"prompt": prompt}
            json_body_prompt_completion = {"prompt": prompt + completion}
        elif url.endswith("/chat"):
            count_tokens_url = url + "/count_tokens"
            json_body_prompt = {"chat_context": chat_context}
            json_body_prompt_completion = {
                "chat_context": chat_context
                + [{"role": "assistant", "content": completion}]
            }
        else:
            raise ValueError(
                "NeMo LLM Service currently supports two LLM endpoints, /completions and /chat"
            )

        count_tokens_headers = dict(headers)
        # we don't need x-stream in our headers
        if "x-stream" in count_tokens_headers:
            count_tokens_headers.pop("x-stream")

        with httpx.Client(timeout=self._get_timeout()) as client:
            # First we just want to get the number of tokens
            # in the prompt itself. Note this includes
            # virtual tokens that are present in the model.
            response = client.post(
                url=count_tokens_url,
                headers=count_tokens_headers,
                json=json_body_prompt,
            )
            response.raise_for_status()
            num_prompt_tokens = response.json()["input_length"]
            self.prompt_tokens += num_prompt_tokens

            # Second we want to get the number of tokens
            # in the completion. Since we don't want to double
            # count the virtual tokens, we include the prompt
            # then subtract num_prompt_tokens.
            response = client.post(
                url=count_tokens_url,
                headers=count_tokens_headers,
                json=json_body_prompt_completion,
            )
            response.raise_for_status()
            num_completion_tokens = response.json()["input_length"] - num_prompt_tokens
            self.completion_tokens += num_completion_tokens

            # total tokens is prompt tokens plus completion tokens
            self.total_tokens += num_prompt_tokens + num_completion_tokens

            # increment number of successful requests
            self.successful_requests += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gen in response.generations:
            # with NeMoLLM, we only have a single generation per gen in response.generations
            self._count_tokens(
                url=response.llm_output["url"],
                headers=response.llm_output["headers"],
                model_name=response.llm_output["model_name"],
                prompt=gen[0].generation_info.get("prompt"),
                chat_context=gen[0].generation_info.get("chat_context"),
                completion=gen[0].text,
            )
