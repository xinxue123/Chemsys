from src.bo.reasoner.base import BaseReasoner
from src.llms.local_lm import LocalLMClient


class LocalLMReasoner(BaseReasoner):
    def __init__(self, exp_config_path: str, result_dir: str, model_path: str):
        super().__init__(exp_config_path, result_dir)
        self.client = LocalLMClient(model_path)
