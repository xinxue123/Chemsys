from src.bo.reasoner.base import BaseReasoner
from src.llms.deepseek import DeepSeekClient


class DSReasoner(BaseReasoner):
    def __init__(self, exp_config_path: str, result_dir: str):
        super().__init__(exp_config_path, result_dir)
        self.client = DeepSeekClient()
