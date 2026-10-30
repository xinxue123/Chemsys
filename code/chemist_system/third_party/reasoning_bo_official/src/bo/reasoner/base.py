import json
from ax import Trial, Arm, GeneratorRun, Experiment
from typing import Dict
import os
import re
import glob
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from src.prompts.base import PromptManager
from src.utils.metric import save_trial_data

from src.utils.jsonl import add_to_jsonl, concatenate_jsonl
from src.bo.models import BOModel
from src.config import Config

config = Config()

system_message = "You are an expert at extracting JSON from text."


class BaseReasoner:
    def __init__(self, exp_config_path: str, result_dir: str):
        self.exp_config = self._load_config(exp_config_path)
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        if not self.result_dir.endswith(('/', '\\')):
            self.result_dir += "/"

        self.trial_data_dir = self.result_dir + "trial_data/"
        self.messages_file_path = self.result_dir + "messages.jsonl"
        self.insight_history_file_path = (
            self.result_dir + "insight_history.jsonl"
        )
        self.experiment_analysis_file_path = (
            self.result_dir + "experiment_analysis.jsonl"
        )

        self.prompt_manager = PromptManager()
        self.experiment_analysis = {}
        self.overview = ""
        self.summary = ""
        self.report = ""
        self.keywords = ""

        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            api_key=config.QWQ_API_KEY,
            url=config.QWQ_API_BASE,
            model_type=config.NOTES_AGENT,
            model_config_dict={"temperature": 0, "max_tokens": 2048},
        )

        self.chat_agent = ChatAgent(
            model=self.model,
            system_message=system_message,
        )

    def _get_system_prompt_template(self, raw_insight: str) -> str:
        return f"""The following text contains a JSON object that may be wrapped in markdown or other formatting. 
        Please extract just the JSON object and ensure it is properly formatted. 
        Return ONLY the JSON object with no additional text or explanation.

        Text to clean:
        {raw_insight}"""

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_trial_data(self):
        csv_files = glob.glob(os.path.join(self.trial_data_dir, "*.csv"))
        combined_data = []
        for file_path in csv_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_data.append(file.read())
        return "\n".join(combined_data)

    def _save_insight(self, trial_index: int) -> None:
        new_insight = self.client.messages[-1]['content']
        data = {"trial_index": trial_index, "insight": new_insight}
        add_to_jsonl(self.insight_history_file_path, data)

    def _save_messages(self):
        self.client.save_messages(self.messages_file_path)

    def _extract_keywords_from_insight(self, insight: str):
        try:
            insight = insight.strip()
            insight = re.sub(
                r'^```json\s*|\s*```$', '', insight, flags=re.MULTILINE
            )
            insight_data = json.loads(insight)

            if isinstance(insight_data, dict) and "keywords" in insight_data:
                keywords = insight_data["keywords"]
                if isinstance(keywords, str):
                    return keywords.strip()
                elif isinstance(keywords, (list, tuple)):
                    return " ".join(str(k) for k in keywords).strip()

            print("Warning: No valid 'keywords' field found in insight")
            return ""

        except json.JSONDecodeError:
            print("Warning: Failed to parse insight as JSON")
            return ""
        except Exception as e:
            print(f"Warning: Unexpected error extracting keywords - {str(e)}")
            return ""

    def get_keywords(self):
        return self.keywords

    def _extract_candidates_from_insight(self, insight, n: int = 3):
        print("Start extracting candidates array from insight...")
        insight = insight.strip()
        insight = re.sub(
            r'^```json\s*|\s*```$', '', insight, flags=re.MULTILINE
        )

        CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
        insight = json.loads(insight)
        if not isinstance(insight, dict) or "hypotheses" not in insight:
            raise ValueError("Invalid JSON format: missing 'hypotheses' key")

        sorted_hypotheses = sorted(
            insight["hypotheses"],
            key=lambda x: CONFIDENCE_ORDER.get(x["confidence"].lower(), 3),
        )

        candidates = []
        for hyp in sorted_hypotheses:
            if "parameter_sets" not in hyp or not isinstance(
                hyp["parameter_sets"], list
            ):
                continue

            for point in hyp["parameter_sets"]:
                if not isinstance(point, dict):
                    continue
                candidates.append(point)
                if len(candidates) == n:
                    print(f"Done! We have collected {n} candidates.")
                    return candidates
        print(f"Done! We have collected less than {n} candidates.")
        return candidates

    def run_bo_experiment(self, experiment, candidates_array):
        print("Start running BO experiment...")
        candidates = [Arm(parameters=params) for params in candidates_array]
        filtered_generator_run = GeneratorRun(arms=candidates)
        trial = experiment.new_batch_trial(
            generator_run=filtered_generator_run
        )
        trial.run()
        trial.mark_completed()
        print("BO experiment completed.")
        return trial

    def _save_experiment_data(self, experiment, trial: Trial) -> None:
        print("Start saving experiment data...")
        self._save_insight(trial_index=trial.index)
        self._save_messages()
        save_trial_data(
            experiment=experiment, trial=trial, save_dir=self.trial_data_dir
        )
        print("Experiment data saved.")

    def generate_overview(self) -> str:
        try:
            print("Start generating overview...")
            formatted_prompt = self.prompt_manager.format(
                "generate_overview", **self.exp_config
            )
            content, _ = self.client.generate(user_prompt=formatted_prompt)
            self.overview = content
            print(f"Overview generated:\n{content}")
            return content

        except Exception as e:
            print(f"Error generating overview: {e}")
            return ""

    def initial_sampling(self) -> str:
        try:
            print("Start initial sampling...")
            meta_dict = {**self.exp_config, "overview": self.overview}
            formatted_prompt = self.prompt_manager.format(
                "initial_sampling", **meta_dict
            )
            raw_insight, _ = self.client.generate(user_prompt=formatted_prompt)
            chat_agent_prompts = self._get_system_prompt_template(raw_insight)
            insight = self.chat_agent.step(chat_agent_prompts).msg.content
            print(f"Initial sampling process completed:\n{insight}")
            return insight

        except Exception as e:
            print(f"Error during initial sampling: {e}")
            return ""

    def optimization_first_round(self, insight):
        candidates = self._extract_candidates_from_insight(insight)
        self.keywords = self._extract_keywords_from_insight(insight)
        return candidates

    def optimization_loop(
        self,
        experiment: Experiment,
        trial: Trial,
        bo_model: BOModel,
        retrieval_context: str = None,
        n: int = 7,
    ) -> str:
        generator_run_by_bo = bo_model.gen(n=n)
        bo_candidates = [arm.parameters for arm in generator_run_by_bo.arms]

        trial_data = self._load_trial_data()

        with open(self.insight_history_file_path, 'r', encoding='utf-8') as f:
            insight_history = []
            for line_number, line in enumerate(f, 1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    insight_history.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(
                        f"Line {line_number} failed to parse, content: {stripped_line[:50]}..., error: {e.msg}"
                    )

        insight_history = concatenate_jsonl(insight_history)

        condidates_array = []
        try:
            print(f"Start Optimization iteration {trial.index}...")
            meta_dict = {
                **self.exp_config,
                "iteration": trial.index,
                "trial_data": trial_data,
                "insight_history": insight_history,
                "bo_recommendations": bo_candidates,
                "retrieved_context": retrieval_context,
            }
            formatted_prompt = self.prompt_manager.format(
                "optimization_loop", **meta_dict
            )
            raw_insight, _ = self.client.generate(user_prompt=formatted_prompt)
            chat_agent_prompts = self._get_system_prompt_template(raw_insight)
            insight = self.chat_agent.step(chat_agent_prompts).msg.content

            print(
                f"Optimization loop iteration {trial.index} completed:\n{insight}"
            )
            self.keywords = self._extract_keywords_from_insight(insight)
            condidates_array = self._extract_candidates_from_insight(insight)

        except Exception as e:
            print(f"Error during optimization iteration {trial.index}: {e}")
            return ""

        self._save_experiment_data(experiment=experiment, trial=trial)
        return condidates_array

    def _generate_summary(self, trial_data, insight_history):
        print("Start generating summary...")
        meta_dict = {
            **self.exp_config,
            "iteration": len(insight_history),
            "trial_data": trial_data,
            "insight_history": insight_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_summary", **meta_dict
        )
        insight, _ = self.client.generate(user_prompt=formatted_prompt)
        print(f"Experiment summary generated:\n{insight}")
        self.summary = insight
        self._save_messages()
        return insight

    def _generate_report(self, trial_data, insight_history):
        print("Start generating report...")
        meta_dict = {
            **self.exp_config,
            "iteration": len(insight_history),
            "trial_data": trial_data,
            "insight_history": insight_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_report", **meta_dict
        )
        insight, _ = self.client.generate(user_prompt=formatted_prompt)
        print(f"Experiment report generated:\n{insight}")
        self.report = insight
        self._save_messages()
        return insight

    def generate_experiment_analysis(self):
        print("Start generating experiment analysis...")
        file_path = self.result_dir + "experiment_analysis.json"
        trial_data = self._load_trial_data()
        with open(self.insight_history_file_path, 'r', encoding='utf-8') as f:
            insight_history = [json.loads(line) for line in f]

        insight_history = concatenate_jsonl(insight_history)
        analysis = {
            "overview": self.overview,
            "summary": self._generate_summary(trial_data, insight_history),
            "report": self._generate_report(trial_data, insight_history),
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)
        print("Experiment analysis generated.")
