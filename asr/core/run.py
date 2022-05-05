from pathlib import Path
from htwutil.run import run_task
from asr.core.repository import ASRRepository


def run_task_in_cwd():
    directory = Path.cwd()
    repo = ASRRepository.find(directory)
    run_task(repo, directory)


if __name__ == '__main__':
    run_task_in_cwd()
