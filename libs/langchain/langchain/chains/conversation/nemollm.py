from langchain.pydantic_v1 import Field

from langchain import ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory, BasePromptTemplate


nemo_user_assistant_template = """\
The following is a friendly conversation between a human User and an AI Assistant. \
The AI is talkative and provides lots of specific details from its context. \
If the AI does not know the answer to a question, it truthfully says it does not know.

{history}
User: {input}"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=nemo_user_assistant_template
)


class NeMoConversationBufferMemory(ConversationBufferMemory):
    human_prefix: str = "User"
    ai_prefix: str = "Assistant"


class NeMoConversationChain(ConversationChain):
    memory: BaseMemory = Field(default_factory=NeMoConversationBufferMemory)
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""
