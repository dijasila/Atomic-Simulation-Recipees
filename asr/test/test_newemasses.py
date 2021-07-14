import pytest
from pytest import approx
import numpy as np
from asr.newemasses import BandFit, BT, make_bandfit, BandstructureData, make_bs_data
from ase.units import Bohr, Ha

@pytest.mark.ci
def test_fitting_retries():
    # Test that if fitting produces wrong sign first
    # the fitting reruns with a larger erange
    from asr.newemasses import perform_fit
    bandfit = make_bandfit(np.zeros((10, 3)), np.zeros(10), BT.cb, (0, 0), 2)
    def fitting_fnc(k0_index, kpts_kv, eps_k, erange, ndim):
        if erange == 1:
            return [-1, -1], None, None
        elif erange == 2:
            return [1, -1], None, None
        else:
            return [1, 1], None, None
            
    eranges = [1, 2, 3]

    perform_fit(bandfit, fitting_fnc=fitting_fnc, eranges=eranges)
    
    assert bandfit.erange == 3


@pytest.mark.ci
def test_fitting_1d():
    from asr.newemasses import polynomial_fit

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for ix in [0, 1, 2]:
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, ix] = kx
        
        eps_k = 0.5 * kx**2
    
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts // 2, kpts_kv, eps_k, 0.5, 1)
        assert len(mass_n) == 1
        assert dir_vn.shape == (3, 1)
        assert np.allclose(mass_n[0], 1.0)
        dir_vn *= np.sign(dir_vn[np.argmax(np.abs(dir_vn[:, 0])), 0])
        assert np.allclose(dir_vn[:, 0], np.array([ix == 0, ix == 1, ix == 2]).astype(float))
    

@pytest.mark.ci
def test_fitting_2d():
    from asr.newemasses import polynomial_fit
    from itertools import product

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for (ix, iy) in product([0, 1, 2], [0, 1, 2]):
        if ix == iy:
            continue
        kpts_kv = np.zeros((npts * npts, 3))
        KX_xy, KY_xy = np.meshgrid(kx, kx, indexing="ij")
        KX_k = KX_xy.reshape(-1)
        KY_k = KY_xy.reshape(-1)
        kpts_kv[:, ix] = KX_k
        kpts_kv[:, iy] = KY_k
        
        eps_k = 0.5 * KX_k**2 + 0.5 * KY_k**2

    
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts * npts // 2, kpts_kv, eps_k, 0.5, 2)
        assert len(mass_n) == 2
        assert dir_vn.shape == (3, 2)
        assert np.allclose(mass_n, 1.0)
        assert np.allclose(abs(dir_vn[ix, 0]), 1.0) or np.allclose(abs(dir_vn[iy, 0]), 1.0)
        if np.allclose(dir_vn[ix, 0], 1.0):
            assert np.allclose(abs(dir_vn[iy, 1]), 1.0)
        else:
            assert np.allclose(abs(dir_vn[ix, 1]), 1.0)

@pytest.mark.ci
def test_fitting_3d():
    from asr.newemasses import polynomial_fit
    ndim = 3
    for i in range(100):
        mx = np.random.rand()*10 + 0.5
        my = np.random.rand()*10 + 0.5
        mz = np.random.rand()*10 + 0.5
        assert not np.allclose(mx, my)
        assert not np.allclose(mx, mz)
        assert not np.allclose(my, mz)

        npts = 11
        kx = np.linspace(-1, 1, npts)
        kpts_kv = np.zeros((npts**ndim, 3))
        KX_xyz, KY_xyz, KZ_xyz = np.meshgrid(kx, kx, kx, indexing="ij")
        KX_k = KX_xyz.reshape(-1)
        KY_k = KY_xyz.reshape(-1)
        KZ_k = KZ_xyz.reshape(-1)
        kpts_kv[:, 0] = KX_k
        kpts_kv[:, 1] = KY_k
        kpts_kv[:, 2] = KZ_k
        
        eps_k = 0.5 / mx * KX_k**2 + 0.5 / my * KY_k**2 + 0.5 / mz * KZ_k**2
        
        
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts * npts // 2, kpts_kv, eps_k, 0.5, ndim)
        assert len(mass_n) == ndim
        assert dir_vn.shape == (3, ndim)
        assert np.allclose(np.sort(mass_n), np.sort(np.array([mx, my, mz])))

        for i in range(0, ndim):
            if np.allclose(mass_n[i], mx):
                assert np.allclose(dir_vn[:, i], np.array([1, 0, 0]))
            if np.allclose(mass_n[i], my):
                assert np.allclose(dir_vn[:, i], np.array([0, 1, 0]))
            if np.allclose(mass_n[i], mz):
                assert np.allclose(dir_vn[:, i], np.array([0, 0, 1]))


@pytest.mark.ci
def test_fitting_3d_complex_dir():
    from asr.newemasses import polynomial_fit
    ndim = 3
    
    for i in range(100):
        mx = np.random.rand()*10 + 0.5
        my = np.random.rand()*10 + 0.5
        mz = np.random.rand()*10 + 0.5
        assert not np.allclose(mx, my)
        assert not np.allclose(mx, mz)
        assert not np.allclose(my, mz)
        npts = 11
        kx = np.linspace(-1, 1, npts)
        kpts_kv = np.zeros((npts**ndim, 3))
        KX_xyz, KY_xyz, KZ_xyz = np.meshgrid(kx, kx, kx, indexing="ij")
        KX_k = KX_xyz.reshape(-1)
        KY_k = KY_xyz.reshape(-1)
        KZ_k = KZ_xyz.reshape(-1)
        kpts_kv[:, 0] = KX_k
        kpts_kv[:, 1] = KY_k
        kpts_kv[:, 2] = KZ_k
        
        eps_k = 0.5 / mx * ((KX_k + KY_k) / np.sqrt(2.0))**2 + 0.5 / my * ((KX_k - KY_k) / np.sqrt(2.0))**2 + 0.5 / mz * KZ_k**2

    
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts * npts // 2, kpts_kv, eps_k, 0.5, ndim)
        assert len(mass_n) == ndim
        assert dir_vn.shape == (3, ndim)
        assert np.allclose(np.sort(mass_n), np.sort(np.array([mx, my, mz])))
        dirs = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                         [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
                         [0.0, 0.0, 1.0]])
        print("complex dir", dir_vn)
        for i in range(0, ndim):
            if np.allclose(mass_n[i], mx):
                assert np.allclose(dir_vn[:, i], dirs[0]) or np.allclose(-dir_vn[:, i], dirs[0])
            if np.allclose(mass_n[i], my):
                assert np.allclose(dir_vn[:, i], dirs[1]) or np.allclose(-dir_vn[:, i], dirs[1])
            if np.allclose(mass_n[i], mz):
                assert np.allclose(dir_vn[:, i], dirs[2]) or np.allclose(-dir_vn[:, i], dirs[2])
    
            
@pytest.mark.ci
def test_shifted_fitting_1d():
    from asr.newemasses import polynomial_fit

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for i in range(50):
        m = np.random.rand() * 10 + 0.05
        shift = np.random.rand() * 1.6 - 0.8
        for ix in [0, 1, 2]:
            kpts_kv = np.zeros((npts, 3))
            kpts_kv[:, ix] = kx
            
            eps_k = 0.5 * (kx - shift)**2 / m
            
            erange = np.max(eps_k) - np.min(eps_k)
            mass_n, dir_vn, lstsq_solution = polynomial_fit(npts // 2, kpts_kv, eps_k, erange, 1)
            assert len(mass_n) == 1
            assert dir_vn.shape == (3, 1)
            assert np.allclose(mass_n[0], m)
            dir_vn *= np.sign(dir_vn[np.argmax(np.abs(dir_vn[:, 0])), 0])
            assert np.allclose(dir_vn[:, 0], np.array([ix == 0, ix == 1, ix == 2]).astype(float))
            assert np.allclose(lstsq_solution[-1], shift**2 * 0.5 / m)
            assert np.allclose(lstsq_solution[-4 + ix], -shift / m)


@pytest.mark.ci
def test_very_anisotropic_2d_band():
    from asr.newemasses import polynomial_fit
    from itertools import product

    npts = 11
    kx = np.linspace(-1, 1, npts)
    mx = np.random.rand() * 10 + 0.01
    my = 1000 * mx
    for (ix, iy) in product([0, 1, 2], [0, 1, 2]):
        if ix == iy:
            continue
        kpts_kv = np.zeros((npts * npts, 3))
        KX_xy, KY_xy = np.meshgrid(kx, kx, indexing="ij")
        KX_k = KX_xy.reshape(-1)
        KY_k = KY_xy.reshape(-1)
        kpts_kv[:, ix] = KX_k
        kpts_kv[:, iy] = KY_k
        
        eps_k = 0.5 * KX_k**2 / mx + 0.5 * KY_k**2 / my

    
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts * npts // 2, kpts_kv, eps_k, 0.5, 2)
        assert len(mass_n) == 2
        assert dir_vn.shape == (3, 2)
        assert np.allclose(np.sort(mass_n), np.array([mx, my]))
        assert np.allclose(abs(dir_vn[ix, 0]), 1.0) or np.allclose(abs(dir_vn[iy, 0]), 1.0)
        if np.allclose(dir_vn[ix, 0], 1.0):
            assert np.allclose(abs(dir_vn[iy, 1]), 1.0)
        else:
            assert np.allclose(abs(dir_vn[ix, 1]), 1.0)



@pytest.mark.ci
def test_extremely_anisotropic_2d_band():
    from asr.newemasses import polynomial_fit
    from itertools import product

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for i in range(50):
        mx = np.random.rand() * 10 + 0.01
        my = 100000 * mx
        for (ix, iy) in product([0, 1, 2], [0, 1, 2]):
            if ix == iy:
                continue
            kpts_kv = np.zeros((npts * npts, 3))
            KX_xy, KY_xy = np.meshgrid(kx, kx, indexing="ij")
            KX_k = KX_xy.reshape(-1)
            KY_k = KY_xy.reshape(-1)
            kpts_kv[:, ix] = KX_k
            kpts_kv[:, iy] = KY_k
            
            eps_k = 0.5 * KX_k**2 / mx + 0.5 * KY_k**2 / my
    
            erange = np.max(eps_k) - np.min(eps_k)
            mass_n, dir_vn, lstsq_solution = polynomial_fit(npts * npts // 2, kpts_kv, eps_k, erange, 2)
            assert len(mass_n) == 2
            assert dir_vn.shape == (3, 2)
            assert np.allclose(np.sort(mass_n), np.array([mx, my]))
            assert np.allclose(abs(dir_vn[ix, 0]), 1.0) or np.allclose(abs(dir_vn[iy, 0]), 1.0)
            if np.allclose(dir_vn[ix, 0], 1.0):
                assert np.allclose(abs(dir_vn[iy, 1]), 1.0)
            else:
                assert np.allclose(abs(dir_vn[ix, 1]), 1.0)


# @pytest.mark.ci
def illustrate_rashba_fromAsClS():
    """Test a Rashba split band that failed for the old emasses."""
    from asr.newemasses import polynomial_fit
    from ase.units import Hartree, Bohr
    import matplotlib.pyplot as plt

    def run_fit(erange, npts, ax):
        m = 1.0
        krange = np.sqrt(2 * m * erange / Hartree)
        ks = np.linspace(-1, 1, npts)
        ks *= krange
        ks /= Bohr
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, 2] = ks
        
        a = 1.0 / (0.5**2) * 0.03 / Bohr**2 / Hartree
        eps_kn = np.zeros((len(ks), 2))
        eps_kn[:, 0] = a * (ks + 0.015)**2
        eps_kn[:, 1] = a * (ks - 0.015)**2        
        eps_kn.sort(axis=1)

        mass_n, dir_vn, sol = polynomial_fit(np.argmin(eps_kn[:, 0]),
                                             kpts_kv, eps_kn[:, 0], 1e-3/Hartree,
                                             1)

        ax.plot(ks, eps_kn[:, 0], label="blup1")
        ax.plot(ks, eps_kn[:, 1], label="blup1")
        
        ax.plot(ks, sol[2] * ks**2 + sol[-2] * ks + sol[-1], label=str(npts))

        return mass_n

    fig, axes = plt.subplots(ncols=2)
    old_mass = run_fit(1e-3, 9, axes[0])
    new_mass = run_fit(25e-3, 50, axes[1])
    print(old_mass, "          ", new_mass)
    plt.show()

if __name__ == "__main__":
    illustrate_rashba_fromAsClS()    

@pytest.mark.ci
def test_simple_rashba_fromAsClS():
    """Test a Rashba split band that failed for the old emasses."""
    from asr.newemasses import polynomial_fit
    from ase.units import Hartree, Bohr
    import matplotlib.pyplot as plt

    def run_fit(erange, npts):
        m = 1.0
        krange = np.sqrt(2 * m * erange / Hartree)
        ks = np.linspace(-1, 1, npts)
        ks *= krange
        ks /= Bohr
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, 2] = ks
        
        a = 1.0 / (0.5**2) * 0.03 / Bohr**2 / Hartree
        eps_kn = np.zeros((len(ks), 2))
        eps_kn[:, 0] = a * (ks + 0.015)**2
        eps_kn[:, 1] = a * (ks - 0.015)**2
        eps_kn.sort(axis=1)

        mass_n, dir_vn, sol = polynomial_fit(np.argmin(eps_kn[:, 0]),
                                             kpts_kv, eps_kn[:, 0], 1e-3/Hartree,
                                             1)

        return mass_n

    old_mass = run_fit(1e-3, 9)
    new_mass = run_fit(25e-3, 50)
    
    assert old_mass < 0.0
    assert new_mass > 0.0


@pytest.mark.ci
def test_adaptive_rashba_fromAsClS():
    """Test a Rashba split band that failed for the old emasses."""
    from asr.newemasses import polynomial_fit, perform_fit, FittingError
    from ase.units import Hartree, Bohr
    import matplotlib.pyplot as plt
    erange = 25e-3
    npts = 9 * 25
    m = 1.0
    krange = np.sqrt(2 * m * erange / Hartree)
    ks = np.linspace(-1, 1, npts)
    ks *= krange
    ks /= Bohr
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 2] = ks
    
    a = 1.0 / (0.5**2) * 0.03 / Bohr**2 / Hartree
    eps_kn = np.zeros((len(ks), 2))
    eps_kn[:, 0] = a * (ks + 0.015)**2
    eps_kn[:, 1] = a * (ks - 0.015)**2
        
    eps_kn.sort(axis=1)
    
    bf1 = make_bandfit(kpts_kv, eps_kn[:, 0], BT.cb, (0, np.argmin(eps_kn[:, 0])), 1)
    lim_inds = np.linspace(-4, 4, 9).astype(int) + npts // 2
    bf2 = make_bandfit(kpts_kv[lim_inds], eps_kn[lim_inds, 0], BT.cb, (0, np.argmin(eps_kn[lim_inds, 0])), 1)

    def fit_func(a, b, c, d, e):
        # Define this function so we are sure
        # we never have too few pts inside
        # polynomial_fit
        x, y, z = polynomial_fit(a, b, c, d, e)
        if y is None:
            raise ValueError()
        return x, y, z

    perform_fit(bf1, fitting_fnc=fit_func, eranges=[1e-3, 25e-3])
    assert np.allclose(bf1.erange, 1e-3)
    try:
        perform_fit(bf2, fitting_fnc=fit_func, eranges=[1e-3, 25e-3])
        raise ValueError(f'Should have failed! {bf2.erange}')
    except FittingError:
        pass
    perform_fit(bf1, fitting_fnc=fit_func, eranges=[1e-8, 1e-3])
    assert np.allclose(bf1.erange, 1e-3)
    


@pytest.mark.ci
def test_toosmall_range():
    """Test that polynomial_fit fails if it gets too few points."""
    from asr.newemasses import polynomial_fit

    for ndim in range(1, 4):
        npts = 2 * ndim
        kx = np.linspace(-1, 1, npts)
        m = np.random.rand() * 10 + 0.05
        shift = np.random.rand() * 1.6 - 0.8
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, 0] = kx
        
        eps_k = 0.5 * (kx - shift)**2 / m
        
        erange = np.max(eps_k) - np.min(eps_k)
        mass_n, dir_vn, lstsq_solution = polynomial_fit(npts // 2, kpts_kv, eps_k, erange, ndim)
        assert np.isnan(mass_n).all()
        assert dir_vn is None
        assert lstsq_solution is None
    

@pytest.mark.ci
def test_veryflatinsmallrange_fit():
    """Test that perform_fit switches to larger range if the small range is unphysical."""
    # Make a band that rapidly changes energy after a flat region
    from asr.newemasses import perform_fit

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    eps_k = np.zeros(npts)
    
    k0_index = npts // 2
    delta_index = 3
    eps_k[:k0_index-delta_index] = 10.0e-3 / Ha * kx[:k0_index-delta_index]**2 + 10e-3 / Ha
    eps_k[k0_index+delta_index:] = 10.0e-3 / Ha * kx[k0_index+delta_index:]**2 + 10e-3 / Ha
    eranges = [1e-3 / Ha, 20e-3 / Ha]

    bf = make_bandfit(kpts_kv, eps_k, BT.cb, 0, 1)
    bf.k0_index = k0_index

    perform_fit(bf, eranges=eranges)

    assert np.allclose(bf.erange, eranges[1])


@pytest.mark.ci
def test_rashba_not_enough_pts():
    """Test that perform_fit switches to larger range if the small range is unphysical.

    Test on Rashba-type band."""    
    # Make a x^4 - x^2 band
    from asr.newemasses import perform_fit, polynomial_fit

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    eps_k = np.zeros(npts)
    
    k0_index = npts // 2
    delta_index = 3
    midks = kx[k0_index-delta_index:k0_index+delta_index]
    eps_k[k0_index-delta_index:k0_index+delta_index] = (midks**4 - midks**2) * 1e-3 / Ha
    eps_k[:k0_index-delta_index] = 10.0e-3 / Ha * kx[:k0_index-delta_index]**2 + 10e-3 / Ha
    eps_k[k0_index+delta_index:] = 10.0e-3 / Ha * kx[k0_index+delta_index:]**2 + 10e-3 / Ha
    eranges = [1e-3 / Ha, 20e-3 / Ha]

    bf = make_bandfit(kpts_kv, eps_k, BT.cb, 0, 1)
    bf.k0_index = k0_index

    masses, _, _ = polynomial_fit(bf.k0_index, bf.kpts_kv, bf.eps_k,
                                  1e-3 / Ha, bf.ndim)

    assert len(masses) == 1
    assert masses[0] < 0.0

    perform_fit(bf, eranges=eranges)

    assert np.allclose(bf.erange, eranges[1])


@pytest.mark.ci
def test_get_be_indices():
    from asr.newemasses import get_be_indices

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for ix in [0, 1, 2]:
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, ix] = kx
        
        eps_skn = np.zeros((1, len(kx), 3))
        eps_skn[0, :, 0] = 0.5 * kx**2
        eps_skn[0, :, 1] = 0.5 * kx**2 + 1
        eps_skn[0, :, 2] = 0.5 * kx**2 + 2

        kmin = np.argmin(eps_skn[0, :, 0])

        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 0, BT.cb, 0.1)
        assert e_ik.shape[0] == 1
        assert np.allclose(sn_i, [(0, 0)])
        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 0, BT.cb, 1)
        assert e_ik.shape[0] == 2
        assert np.allclose(sn_i, [(0, 0), (0, 1)])
        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 0, BT.cb, 2)
        assert e_ik.shape[0] == 3
        assert np.allclose(sn_i, [(0, 0), (0, 1), (0, 2)])


@pytest.mark.ci
def test_get_be_indices_vb():
    from asr.newemasses import get_be_indices

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for ix in [0, 1, 2]:
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, ix] = kx
        
        eps_skn = np.zeros((1, len(kx), 3))
        eps_skn[0, :, 0] = -0.5 * kx**2
        eps_skn[0, :, 1] = -0.5 * kx**2 + 1
        eps_skn[0, :, 2] = -0.5 * kx**2 + 2

        kmin = np.argmin(eps_skn[0, :, 0])

        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 2, BT.vb, 0.1)
        assert e_ik.shape[0] == 1
        assert np.allclose(sn_i, [(0, 2)])
        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 2, BT.vb, 1)
        assert e_ik.shape[0] == 2
        assert np.allclose(sn_i, [(0, 1), (0, 2)])
        e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 2, BT.vb, 2)
        assert e_ik.shape[0] == 3
        assert np.allclose(sn_i, [(0, 0), (0, 1), (0, 2)])


@pytest.mark.ci
def test_get_be_indices_fails():
    from asr.newemasses import get_be_indices

    npts = 11
    kx = np.linspace(-1, 1, npts)
    for ix in [0, 1, 2]:
        kpts_kv = np.zeros((npts, 3))
        kpts_kv[:, ix] = kx
        
        eps_skn = np.zeros((1, len(kx), 3))
        eps_skn[0, :, 0] = -0.5 * kx**2
        eps_skn[0, :, 1] = -0.5 * kx**2 + 1
        eps_skn[0, :, 2] = -0.5 * kx**2 + 2

        kmin = np.argmin(eps_skn[0, :, 0])
        
        try:
            e_ik, sn_i = get_be_indices(eps_skn, 0, kmin, 2, None, 0.1)
            assert False, "Did not fail when it was expected to!"
        except :
            pass


@pytest.mark.ci
def test_serialization_of_bandfit():

    # Basic data serialization
    bf = make_bandfit(np.random.rand(10, 3), np.random.rand(10), BT.cb, (0, 5), 1)
    bf2 = BandFit.from_dict(bf.to_dict())
    
    assert np.allclose(bf.kpts_kv, bf2.kpts_kv)
    assert np.allclose(bf.eps_k, bf2.eps_k)
    assert bf.bt == bf2.bt
    assert bf.band == bf2.band
    assert bf.ndim == bf2.ndim

def compare_bsd(bsd, bsd2):
    b1 = np.allclose(bsd.kpts_kc, bsd2.kpts_kc)
    b2 = np.allclose(bsd.e_k, bsd2.e_k)
    b3 = np.allclose(bsd.spin_k, bsd2.spin_k)
    b4 = np.allclose(bsd.kpts_kv, bsd2.kpts_kv)
    return b1 and b2 and b3


@pytest.mark.ci
def test_serialization_of_bsdata():
    bsd = make_bs_data(np.random.rand(10, 3), np.random.rand(10, 3),
                       np.random.rand(10), np.random.rand(10))
    bsd2 = BandstructureData.from_dict(bsd.to_dict())
    assert compare_bsd(bsd, bsd2)


@pytest.mark.ci
def test_serialization_of_full_bandfit():
    # Full data serialization
    random_bsd = lambda : make_bs_data(np.random.rand(10, 3), np.random.rand(10, 3),
                                       np.random.rand(10), np.random.rand(10))
    bf = BandFit(0, np.random.rand(10, 3), np.random.rand(10), BT.cb, (0, 5), 3,
                 mass_n=np.random.rand(3), dir_vn=np.random.rand(3, 3), erange=1e-2,
                 fit_params=np.random.rand(10),
                 bs_data=[random_bsd(), random_bsd(), random_bsd()],
                 bs_erange=None, bs_npoints=None)
    bf2 = BandFit.from_dict(bf.to_dict())
    
    assert np.allclose(bf.kpts_kv, bf2.kpts_kv)
    assert np.allclose(bf.eps_k, bf2.eps_k)
    assert bf.bt == bf2.bt
    assert bf.band == bf2.band
    assert bf.ndim == bf2.ndim

    assert np.allclose(bf.mass_n, bf2.mass_n)
    assert np.allclose(bf.dir_vn, bf2.dir_vn)
    assert np.allclose(bf.erange, bf2.erange)
    assert np.allclose(bf.fit_params, bf2.fit_params)
    assert all(compare_bsd(x, y) for x, y in zip(bf.bs_data, bf2.bs_data))



"""
We want to test calc_bandstructure somehow.
But it does a gpw calc, so we need a way to mock that
Also other stuff need to be mocked
"""

@pytest.mark.ci
def test_calc_bandstructure():
    """Test that bandstructure is constructed correctly from returned data.
    
    This step is essential to the process so we need to ensure it works,
    even though it is relatively trivial.
    """
    from asr.newemasses import calc_bandstructure

    kpts = np.random.rand(10, 3)
    band = 6
    bf = make_bandfit(kpts, np.random.rand(10), BT.vb, band, 1)

    class MockCalc:
        def get_bz_k_points(self):
            return kpts
    
    def createcalc(baf, dre, ca):
        return MockCalc(), None

    def spinaxis():
        return 0, 0
        
    e_km, blup, s_kvm = np.random.rand(10, band * 2), None, np.random.rand(10, 3, band * 2)
    def eigscalc(calc, soc, return_spin, theta, phi):
        return e_km, blup, s_kvm

    def spinindex():
        return 2

    calc_bandstructure(bf, None,
                       createcalc_fn=createcalc,
                       eigscalc_fn=eigscalc,
                       spinaxis_fn=spinaxis,
                       spinindex_fn=spinindex)

    assert len(bf.bs_data) == 1
    assert np.allclose(bf.bs_data[0].kpts_kc, kpts)
    assert np.allclose(bf.bs_data[0].e_k, e_km[:, band])
    assert np.allclose(bf.bs_data[0].spin_k, s_kvm[:, 2, band])


# Can/should we test create_calc? It does a gpaw calculation
# so we need to mock that. Will it get too dirty?
# But it seems the only way we can really test it.
# Perhaps with refactoring we can separate the testable stuff
# i.e. k-points setup from the non-testable.
# Yes, let's write a function to get the kpts
@pytest.mark.ci
def test_kpts_for_bs():
    """Test that kpoints look correct.

    Includes:
    - kpoints are zero in non-periodic directions
    - ??
    """
    from asr.newemasses import get_kpts_for_bandstructure

    periodic_index = int(np.random.rand() * 3)
    pbc = np.arange(0, 3) == periodic_index

    
    kpts_kv = np.zeros((10, 3))
    kpts_kv[:, periodic_index] = np.random.rand(10)
    band = 6
    ndim = 1
    bf = make_bandfit(kpts_kv, np.random.rand(10), BT.vb, band, ndim)
    direction = 0
    bf.mass_n = np.array([1.0] * (ndim))


    bf.dir_vn = np.zeros((3, ndim))
    bf.dir_vn[:, direction] = np.array([1.0, 2.0, 3.0]) * (np.arange(0, 3) == periodic_index)

    bs_erange = 250e-3 / Ha
    bs_npoints = 10
    cell_cv = np.eye(3)

    
    k_kc, _ = get_kpts_for_bandstructure(bf, direction, bs_erange, bs_npoints,
                                         cell_cv, pbc)

    for i in range(ndim):
        if not pbc[i]:
            assert np.allclose(k_kc[:, i], 0.0)

    assert not np.allclose(k_kc, 0.0)
    

@pytest.mark.ci
def test_validation_mae():
    from asr.newemasses import calc_errors

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    e_k = 0.5 * kx**2 + 0.1
    
    error = 1.0
    fit_e_k = e_k + error

    maes, mares = calc_errors(BT.cb, kpts_kv, e_k, fit_e_k, [e_k[0] - e_k[11]])

    assert len(maes) == 1
    assert len(mares) == 1

    assert np.allclose(maes[0, 1], error)


@pytest.mark.ci
def test_validation_mare():
    from asr.newemasses import calc_errors

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    e_k = 0.5 * kx**2 + 1
    
    rerror = 0.1
    fit_e_k = e_k * (1.0 + rerror)

    erange = e_k[0] - e_k[11]
    indices = np.where(np.abs(e_k - np.min(e_k)) < erange)
    mean_de = np.mean(np.abs(e_k[indices] - np.min(e_k[indices])))
    expected_error = np.mean(np.abs((fit_e_k[indices] - e_k[indices])/mean_de))

    maes, mares = calc_errors(BT.cb, kpts_kv, e_k, fit_e_k, [e_k[0] - e_k[11]])

    assert len(maes) == 1
    assert len(mares) == 1

    assert np.allclose(mares[0, 1], expected_error)


@pytest.mark.ci
def test_integration_cb():
    """Run a test of fitting, bandstructure calc, validation."""
    from asr.newemasses import perform_fit, calc_bandstructure, calc_parabolicities
    from ase.dft.kpoints import kpoint_convert

    cell_cv = np.eye(3) * 10.0
    nbands = 2

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    e_k = 0.5 * kx**2 + 1

    bf = make_bandfit(kpts_kv, e_k, BT.cb, 0, 1)

    perform_fit(bf)

    class MockCalc:
        def get_bz_k_points(self):
            return kpoint_convert(cell_cv, ckpts_kv=kpts_kv)

    def createcalc(bf, direction, calc):
        return MockCalc(), kpts_kv

    def spinaxis():
        return 0.0, 0.0

    def eigscalc(calc, soc, return_spin, theta, phi):
        # Return energies in eV
        # Return blup
        # Return spin

        s_kvm = np.zeros((npts, 3, nbands))

        return np.array([e_k * Ha]).T, None, s_kvm

    def spinindex():
        return 0

    calc_bandstructure(bf, None, createcalc, eigscalc, spinaxis, spinindex)
        
    calc_parabolicities(bandfits=[bf], eranges=[(np.max(e_k) - np.min(e_k)) * Ha])

    assert np.allclose(bf.bs_data[0].maes[:, 1], 0.0)
    assert np.allclose(bf.bs_data[0].mares[:, 1], 0.0)
    

    


@pytest.mark.ci
def test_integration_vb():
    """Run a test of fitting, bandstructure calc, validation."""
    from asr.newemasses import perform_fit, calc_bandstructure, calc_parabolicities
    from ase.dft.kpoints import kpoint_convert

    cell_cv = np.eye(3) * 10.0
    nbands = 2

    npts = 21
    kx = np.linspace(-1, 1, npts)
    kpts_kv = np.zeros((npts, 3))
    kpts_kv[:, 0] = kx
    e_k = -0.5 * kx**2 - 1

    bf = make_bandfit(kpts_kv, e_k, BT.vb, 0, 1)

    perform_fit(bf)


    class MockCalc:
        def get_bz_k_points(self):
            return kpoint_convert(cell_cv, ckpts_kv=kpts_kv)

    def createcalc(bf, direction, calc):
        return MockCalc(), kpts_kv

    def spinaxis():
        return 0.0, 0.0

    def eigscalc(calc, soc, return_spin, theta, phi):
        # Return energies in eV
        # Return blup
        # Return spin

        s_kvm = np.zeros((npts, 3, nbands))

        return np.array([e_k * Ha]).T, None, s_kvm

    def spinindex():
        return 0

    calc_bandstructure(bf, None, createcalc, eigscalc, spinaxis, spinindex)

    calc_parabolicities([bf], eranges=[(np.max(e_k) - np.min(e_k)) * Ha])

    assert np.allclose(bf.bs_data[0].maes[:, 1], 0.0)
    assert np.allclose(bf.bs_data[0].mares[:, 1], 0.0)
    
