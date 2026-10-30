from openai import OpenAI
from typing import Tuple
from src.config import Config
from src.utils.jsonl import add_to_jsonl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

config = Config()


class LocalLMClient:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.messages = []

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Tuple[str, str]:
        self.messages = [{"role": "user", "content": user_prompt}]

        inputs = self.tokenizer(user_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        self.messages.append({"role": "assistant", "content": response})
        return response, ""

    def view_messages(self):
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

    def save_messages(self, file_path):
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
