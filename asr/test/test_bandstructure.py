import pytest
from ase.io import write
from asr.c2db.gs import GSWorkflow
from asr.c2db.bandstructure import BSWorkflow


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

    with repo:
        gsw = repo.run_workflow(GSWorkflow, atoms=test_material,
                                calculator=fast_calc)

        bsw = repo.run_workflow(
            BSWorkflow,
            gsworkflow=gsw,
            kptpath=None,
            bsrestart=bsrestart,
            npoints=npoints)

        #tree = repo.tree()

        #futures = list(bsw.postprocess.ancestors())

    bsw.postprocess.runall_blocking(repo)

    # XXX something about locking
    #for future in futures:
    #    future.run_blocking(repo)

    with repo:
        res = bsw.postprocess.value().output

        assert len(res.bs_soc['path'].kpts) == npoints
        assert len(res.bs_nosoc['path'].kpts) == npoints

    write('structure.json', test_material)

    # XXX get webcontent!
    # get_webcontent()
