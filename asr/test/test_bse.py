import pytest

from asr.c2db.gs import GSWorkflow


@pytest.mark.ci
def test_bse(repo,
        asr_tmpdir_w_params, test_material, fast_calc,
        mockgpaw, mocker, get_webcontent):
    from asr.c2db.bse import BSEWorkflow

    import gpaw
    import gpaw.occupations
    from gpaw.response.bse import BSE
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    gpaw.GPAW._get_band_gap.return_value = 1.0
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5
    mocker.patch.object(gpaw.occupations, "FermiDirac")
    gpaw.occupations.FermiDirac.return_value = None

    ndim = sum(test_material.pbc)

    def calculate(self):
        E_B = 1.0

        data = {}
        data['E_B'] = E_B
        data['__key_descriptions__'] = \
            {'E_B': 'KVP: BSE binding energy (Exc. bind. energy) [eV]'}

        return data

    mocker.patch.object(BSE, "calculate", calculate)

    def combined_workflow(rn, atoms, calculator, kptdensity):
        gs_workflow = GSWorkflow(rn, atoms=test_material, calculator=fast_calc)
        bse_workflow = BSEWorkflow(rn, gs_workflow, kptdensity=kptdensity)
        return bse_workflow

    if ndim > 1:
        
        with repo:
            bseworkflow = repo.run_workflow(
                    combined_workflow, test_material, fast_calc, 2)
                    

        #test_material.write("structure.json")
        #get_webcontent()
    else:
        with pytest.raises(NotImplementedError):
            repo.run_workflow_blocking(
                    combined_workflow, test_material, calculator=fast_calc, kptdensity=2)

