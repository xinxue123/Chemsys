"""Use the tested immutable queue in a separate V14 namespace."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from cold_start_v13 import transport as frozen
from cold_start_v13.transport import read, write_once, file_sha, digest
from . import VARIANT


def base(root):
    return (Path(root).resolve()/'experiments/runs'/VARIANT).resolve()


@contextmanager
def scope(root):
    with patch.object(frozen,'base',return_value=base(root)), patch.object(frozen,'VARIANT',VARIANT):
        yield


def request(root,*args,**kwargs):
    with scope(root):
        return frozen.request(root,*args,**kwargs)


def pending(root):
    with scope(root):
        return frozen.pending(root)


def finish(root,agents):
    rows={'/root/'+r['invocation']['task_name']:r for r in pending(root)}
    with scope(root):
        for agent in agents:
            frozen.record_spawn(root,rows[agent]['request'],dict(task_name=agent))
            frozen.seal(root,rows[agent]['request'])
