import importlib
from pathlib import Path
import sys
modules = Path(__file__).parent.glob('./recipies/*.py')
for module in modules:
    name = module.with_suffix('').name
    mod = importlib.import_module('.recipies.' + name, package='mcr')
    sys.modules['mcr.' + name] = mod

