import pytest
from ase.utils import workdir


@pytest.mark.ci
def test_phonons(asr_tmpdir_w_params, mockgpaw, test_material, get_webcontent, fast_calc):
    """Simple test of phonon recipe."""
    from asr.c2db.phonons import calculate, postprocess
    from asr.c2db.gs import GS
    test_material.write('structure.json')

    gs = GS(atoms=test_material, calculator=fast_calc)

    with workdir('calculate', mkdir=True):
        phononresult = calculate(atoms=test_material, calculator=fast_calc,
                                 magstate=gs.magstate)

    # XXX duplicate passing of atoms.  (Also the "n" parameter has this problem)
    # Maybe define a container class for this definition.
    with workdir('post', mkdir=True):
        postprocess(phononresult=phononresult, atoms=test_material)

    # Need to explicitly change paths or the phonon caches will clash

    # main(atoms=test_material)
    # content = get_webcontent()
    # assert "Phonons" in content, content
