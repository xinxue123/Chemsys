"""A majority gate must count distinct registered tasks, never duplicate rows."""
from pathlib import Path
import unittest
from report_knowledge_numeric_loop import registered_run


class RegistrationTests(unittest.TestCase):
    def test_rejects_duplicate_wrong_path_and_unregistered_identity(self):
        folder=Path(__file__).resolve().parents[2]/'runtime/synthetic_v8_report'
        state=dict(task='Case1',mode='full',condition='agent_initial_terra',seed=0)
        protocol=dict(tasks=['Case1'],supported_modes=['bo','full'])
        path=folder/'runs/Case1_full_agent_initial_terra/state.json'
        key=registered_run(state,path,folder,protocol,{})
        self.assertEqual(key,('Case1','full'))
        with self.assertRaises(ValueError):registered_run(state,path,folder,protocol,{key:state})
        with self.assertRaises(ValueError):registered_run(state,path.with_name('copy.json'),folder,protocol,{})
        for changes in (dict(seed=1),dict(condition='correct'),dict(task='missing'),dict(mode='shadow')):
            with self.assertRaises(ValueError):registered_run(dict(state,**changes),path,folder,protocol,{})


if __name__=='__main__':unittest.main()
