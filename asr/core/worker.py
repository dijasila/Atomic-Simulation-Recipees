from htwutil.worker import main
from asr.core.repository import ASRRepository


if __name__ == '__main__':
    repo = ASRRepository.find()
    main(repo)
