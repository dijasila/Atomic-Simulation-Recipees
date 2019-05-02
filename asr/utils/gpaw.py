import os

if not os.environ.get('ASR_TEST_MODE'):
    from gpaw import GPAW, KohnShamConvergenceError
else:
    from asr.utils.emt import GPAW, KohnShamConvergenceError   # noqa: F401
