import pytest
from ase.io import write
from asr.c2db.gs import GSWorkflow
from asr.c2db.bandstructure import workflow as bsworkflow


@pytest.mark.ci
def test_bandstructure_workflow(repo, mockgpaw, test_material,
                                get_webcontent, fast_calc):

    # XXX: Structureinfo is needed for the webpanel to function.
    # This is not really a standard dependency and it should probably
    # be fixed in the future.
    # structinfo(atoms=test_material)
    npoints = 20
    bsrestart = {'nbands': -2,
                 'txt': 'bs.txt',
                 'fixdensity': True}

    gsw = repo.run_workflow(GSWorkflow, atoms=test_material,
                               calculator=fast_calc)

    bs_dct = repo.run_workflow(
        bsworkflow,
        gsresult=gsw.scf.output,
        mag_ani=gsw.magnetic_anisotropy.output,
        gspostprocess=gsw.postprocess.output,
        kptpath=None,
        bsrestart=bsrestart,
        npoints=npoints)

    repo.tree().run_blocking()

    res = bs_dct['postprocess'].value().output

    assert len(res.bs_soc['path'].kpts) == npoints
    assert len(res.bs_nosoc['path'].kpts) == npoints

    write('structure.json', test_material)

    # XXX get webcontent!
    # get_webcontent()
