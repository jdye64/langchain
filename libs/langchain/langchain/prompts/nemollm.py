from typing import Any
from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING


class NeMoFewShotPromptTemplate(FewShotPromptTemplate):
    example_completion: PromptTemplate

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)

        # Just the prompts, not the completions.
        examples_prompts = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format as string
        example_prompt_strings = [
            self.example_prompt.format(**example) for example in examples_prompts
        ]

        # Just the completions, not the prompts
        example_completions = [
            {k: e[k] for k in self.example_completion.input_variables} for e in examples
        ]
        # Format as string
        example_completion_strings = [
            self.example_completion.format(**ec) for ec in example_completions
        ]

        # Alternate between example prompts and completions
        assert len(example_prompt_strings) == len(example_completion_strings)
        prompts_and_completions = []
        for i, (p, c) in enumerate(
            zip(example_prompt_strings, example_completion_strings)
        ):
            if i == 0:
                prompts_and_completions.append(p)
            else:
                prompts_and_completions.append("User: " + p)
            prompts_and_completions.append("Assistant: " + c)

        # Create the overall template.
        pieces = [self.prefix, *prompts_and_completions, "User: " + self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)
