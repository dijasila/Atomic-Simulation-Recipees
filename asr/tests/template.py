# Test a full workflow
import asr
from asr.utils.recipe import Recipe
from pathlib import Path

material = '# Material #'
formula = material.split('.')[0]

asrtestfolder = Path(asr.__path__[0]) / 'tests'
srcstart = Path(asrtestfolder) / material
srcparams = Path(asrtestfolder) / f'{formula}_params.json'
dststart = Path('.') / 'unrelaxed.json'
dstparams = Path('.') / 'params.json'

if not dststart.is_file():
    dststart.write_bytes(srcstart.read_bytes())

if not dstparams.is_file():
    dstparams.write_bytes(srcparams.read_bytes())

# Make sure to reload the module
Recipe.frompath('asr.relax', reload=True).run()
# DepSection #
Recipe.frompath('asr.collect', reload=True).run()
Recipe.frompath('asr.browser', reload=True).run(args=['--only-figures'])
