from typing import List, Union, Optional, Literal
import dataclasses
import os

from openai import OpenAI


# ============================================================
# FIXED CONFIG
# ============================================================

BASE_URL = "https://openrouter.ai/api/v1"
FIXED_MODEL_NAME = "openai/gpt-4o-mini"

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Run: export OPENAI_API_KEY='your_api_key_here'"
    )


MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


def gpt_chat(
    client,
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps: int = 1,
    logprobs: bool = False,
):
    response = client.chat.completions.create(
        model=FIXED_MODEL_NAME,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=logprobs,
    )

    if num_comps == 1:
        return (
            response.choices[0].message.content,
            response.usage,
            response.choices[0].finish_reason,
        )

    return [choice.message.content for choice in response.choices]


class ModelBase:
    """
    Compatibility base class.

    The original repo imports ModelBase from generators.model,
    so we keep this class even though all real calls now go
    through one fixed OpenAI-compatible chat model.
    """

    def __init__(self, name: str):
        self.name = FIXED_MODEL_NAME
        self.is_chat = True
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __repr__(self) -> str:
        return self.name

    def generate_chat(
        self,
        messages: List[Message],
        format_instructions: str = "",
        parser=None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1,
        logprobs: bool = False,
    ) -> dict:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        raise NotImplementedError

    def get_langchain_model(self, temperature: float = 0.0):
        raise NotImplementedError

    def print_usage(self):
        raise NotImplementedError


class GPTChat(ModelBase):
    """
    Single fixed model wrapper.

    The original code can still call:

        GPT35("gpt4-turbo-128k")
        GPT4("gpt4-turbo-128k")
        OpenChat("...")
        StarChat()
        CodeLlama()

    But all of them internally use FIXED_MODEL_NAME.
    """

    def __init__(self, model_name: str = FIXED_MODEL_NAME, *args, **kwargs):
        super().__init__(FIXED_MODEL_NAME)

        self.name = FIXED_MODEL_NAME
        self.is_chat = True

        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    def generate_chat(
        self,
        messages: List[Message],
        format_instructions: str = "",
        parser=None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps: int = 1,
        logprobs: bool = False,
    ) -> dict:

        response = self.client.chat.completions.create(
            model=self.name,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=logprobs,
        )

        if response.usage is not None:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

        output_finish_reason = response.choices[0].finish_reason
        output_text = response.choices[0].message.content

        if parser is None:
            return {
                "parse_success": True,
                "finish_reason": output_finish_reason,
                "raw_response": output_text,
                "output": output_text,
            }

        try:
            output_dict = parser.invoke(output_text)
            output_dict["parse_success"] = True

        except Exception as e:
            retry_messages = messages + [
                Message(
                    role="assistant",
                    content=output_text,
                ),
                Message(
                    role="user",
                    content=(
                        format_instructions
                        + f"\n\nWhen I parse your output, I got this error: {e}"
                    ),
                ),
            ]

            response_2 = self.client.chat.completions.create(
                model=self.name,
                messages=[dataclasses.asdict(message) for message in retry_messages],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=logprobs,
            )

            if response_2.usage is not None:
                self.prompt_tokens += response_2.usage.prompt_tokens
                self.completion_tokens += response_2.usage.completion_tokens

            output_retry_finish_reason = response_2.choices[0].finish_reason
            output_retry_text = response_2.choices[0].message.content

            try:
                output_dict = parser.invoke(output_retry_text)
                output_dict["parse_success"] = True
            except Exception:
                output_dict = {"parse_success": False}

            output_dict["retry_finish_reason"] = output_retry_finish_reason
            output_dict["raw_response"] = output_retry_text

        output_dict["finish_reason"] = output_finish_reason

        if "raw_response" not in output_dict:
            output_dict["raw_response"] = output_text

        return output_dict

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> Union[List[str], str]:

        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            stop=stop_strs,
        )

        if response.usage is not None:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

        if num_comps == 1:
            return response.choices[0].message.content

        return [choice.message.content for choice in response.choices]

    def get_langchain_model(self, temperature: float = 0.0):
        raise NotImplementedError(
            "LangChain support was removed in this simplified fixed-model version."
        )

    def print_usage(self):
        print(
            f"******* {self.name} *******\n"
            f"Prompt tokens: {self.prompt_tokens}\n"
            f"Completion tokens: {self.completion_tokens}\n"
        )


# ============================================================
# Compatibility model classes
# The original repo imports these names from generators.model.
# They all point to the same fixed model.
# ============================================================

class GPT35(GPTChat):
    pass


class GPT4(GPTChat):
    pass


class OpenChat(GPTChat):
    pass


class GPTDavinci(GPTChat):
    pass


class StarChat(GPTChat):
    pass


class CodeLlama(GPTChat):
    pass