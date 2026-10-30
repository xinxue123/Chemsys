import json
import os


class PromptManager:
    """
    A class to manage and format prompt templates stored in JSON files.
    Allows support for multi-language prompts (e.g., 'en', 'zh').
    """

    def __init__(
        self,
        prompt_dir: str = os.path.dirname((os.path.abspath(__file__))),
    ):
        """
        Initialize the TextPrompt class.

        Args:
            prompt_dir: str
                The directory where prompt JSON files are stored.
        """
        self.prompt_dir = prompt_dir
        # print("Prompts registry directory:", self.prompt_dir)
        self.prompts = {}
        self._load_all_prompts()

    def _load_all_prompts(self):
        """
        Load all prompt JSON files from the specified directory.
        """
        # 检查目录是否存在
        if not os.path.exists(self.prompt_dir):
            raise FileNotFoundError(
                f"Prompt directory '{self.prompt_dir}' does not exist."
            )

        for filename in os.listdir(self.prompt_dir):
            if filename.endswith(".json"):  # 只加载后缀为 .json 的文件
                filepath = os.path.join(self.prompt_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        prompt_data = json.load(f)
                        # print(f"Loaded {filename}")
                        # 确保 prompt_data 结构是字典
                        if not isinstance(prompt_data, dict):
                            raise ValueError(
                                f"Invalid structure in {filename}, expected JSON object."
                            )

                        self.prompts.update(prompt_data)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error loading {filename}: {e}")

    def get(self, key: str, lang: str = "en") -> str:
        """
        Retrieve a prompt by key and language.

        Args:
            key: str
                The key of the desired prompt (e.g., "test_prompt").
            lang: str
                The language of the prompt (e.g., "en" for English, "zh" for Chinese).

        Returns:
            str: The prompt template if found, otherwise an empty string.
        """
        return self.prompts.get(key, {}).get(lang, "")

    def format(self, key: str, lang: str = "en", **kwargs) -> str:
        """
        Format a prompt template with provided arguments.

        Args:
            key: str
                The key of the desired prompt.
            lang: str
                The language of the prompt.
            kwargs: dict
                Arguments to customize the template via .format(**kwargs).

        Returns:
            str: The formatted prompt.
        """
        template = self.get(key, lang)
        if not template:
            raise ValueError(
                f"Prompt not found for key '{key}' and language '{lang}'"
            )
        return template.format(**kwargs)
