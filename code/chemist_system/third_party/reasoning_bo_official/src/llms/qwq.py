from openai import OpenAI
from typing import Tuple
from src.config import Config
from src.utils.jsonl import add_to_jsonl

config = Config()


class QWQClient:
    """QWQ client that only supports streaming generation"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.QWQ_API_KEY, base_url=config.QWQ_API_BASE
        )
        self.messages = []
        self.content = ""
        self.reasoning_content = ""

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> Tuple[str, str]:
        """Single round generation using streaming API"""
        self.messages = []
        if json_output:
            system_prompt = "response in JSON format"
            self.messages.append({"role": "system", "content": system_prompt})

        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=config.QWQ_MODEL_NAME,
            messages=self.messages,
            max_tokens=max_tokens,
            stream=True,
        )

        self.content = ""
        self.reasoning_content = ""

        for chunk in response:
            if (
                hasattr(chunk.choices[0].delta, 'reasoning_content')
                and chunk.choices[0].delta.reasoning_content
            ):
                self.reasoning_content += chunk.choices[
                    0
                ].delta.reasoning_content
            if (
                hasattr(chunk.choices[0].delta, 'content')
                and chunk.choices[0].delta.content
            ):
                self.content += chunk.choices[0].delta.content

        self.messages.append(
            {
                "role": "think",
                "content": self.reasoning_content,
            }
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": self.content,
            }
        )
        return (self.content, self.reasoning_content)

    def multi_round_generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
        """Multi-round generation using streaming API"""
        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=config.QWQ_MODEL_NAME,
            messages=self.messages,
            stream=True,
            max_tokens=max_tokens,
        )

        self.content = ""
        self.reasoning_content = ""

        for chunk in response:
            if (
                hasattr(chunk.choices[0].delta, 'reasoning_content')
                and chunk.choices[0].delta.reasoning_content
            ):
                self.reasoning_content += chunk.choices[
                    0
                ].delta.reasoning_content
            if (
                hasattr(chunk.choices[0].delta, 'content')
                and chunk.choices[0].delta.content
            ):
                self.content += chunk.choices[0].delta.content

        self.messages.append(
            {
                "role": "assistant",
                "content": self.content,
            }
        )
        return (self.content, self.reasoning_content)

    def view_messages(self):
        """View message history for debugging purposes"""
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

    def save_messages(self, file_path):
        """
        Save client.messages(list) content as JSONL format with paired user, think, assistant content.
        """
        print("Start saving the message data for this round of trials.\n")
        distill_data = {}
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                distill_data["user"] = content
            elif role == "think":
                distill_data["think"] = content
            elif role == "assistant":
                distill_data["assistant"] = content
        add_to_jsonl(file_path, distill_data)
        print("Save Messages Done!\n")
