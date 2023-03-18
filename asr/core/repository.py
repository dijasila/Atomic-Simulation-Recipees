from htwutil.repository import Repository as HTWRepository
from asr.core.serialize import ASRJSONCodec


class ASRRepository(HTWRepository):
    def __init__(self, root):
        super().__init__(root, usercodec=ASRJSONCodec(),
                         run_module='asr.core.worker')
