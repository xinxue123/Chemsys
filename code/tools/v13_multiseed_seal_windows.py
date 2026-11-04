"""Parent-only Windows path spelling for unchanged immutable response sealing.

The Perera response-receipt temporary pathname has260 characters. Add the
Windows extended-length prefix only at the final filesystem write boundary.
Native canonical bytes, locks, exclusive creation, fsync and replace are kept.
Never used to launch or modify experimental workers.
"""
from pathlib import Path
from unittest.mock import patch
import os
import sys

import v13_multiseed_dispatch as dispatch

transport = dispatch.transport
native_write = transport.write_once
native_write_bytes = transport.write_once_bytes


def filesystem_path(path):
    value = str(Path(path).resolve())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


def write(path, value):
    return native_write(filesystem_path(path), value)


def write_bytes(path, data):
    return native_write_bytes(filesystem_path(path), data)


def main():
    if len(sys.argv) < 2 or any(not arg.startswith('/root/v13_') for arg in sys.argv[1:]):
        raise ValueError('Only actual completed-agent IDs may be sealed by this helper')
    native_write(dispatch.study.directory(dispatch.ROOT) / 'transport_path_adapter.json', dict(
        purpose='Parent_only_extended_Windows_path_spelling_at_immutable_write_boundary',
        helper_sha256=transport.file_sha(__file__), scientific_code_modified=False,
        worker_processes_modified=False, response_bytes_modified=False,
        native_atomic_immutable_writer_preserved=True, failed_temp_path_characters=260))
    sys.argv.insert(1, 'finish')
    with patch.object(transport, 'write_once', write), patch.object(transport, 'write_once_bytes', write_bytes):
        dispatch.main()


if __name__ == '__main__':
    main()
