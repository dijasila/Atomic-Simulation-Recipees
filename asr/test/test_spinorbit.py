import pytest


@pytest.mark.ci
def test_collinear_spinorbit(asr_tmpdir_w_params, test_material, get_webcontent):
    """Test of spinorbit recipe."""
    from asr.spinorbit import calculate, main
    from asr.gs import calculate as gs_calculate
    from ase.parallel import world

    test_material.write('structure.json')
    calculator = {"name": "gpaw",
                  "mode": {"name": "pw", "ecut": 300},
                  "kpts": {"density": 3, "gamma": True},
                  "txt": "gs.txt"}

    gs_calculate(calculator)
    calculate(gpw='gs.gpw', distance=90.0, projected=False, width=0.001)
    result = main()

    assert result.projected_soc is False

    if world.size == 1:
        content = get_webcontent()
        assert "<td>Spinorbitbandwidth(meV)</td>" in content, content
        assert "<td>SpinorbitMinimum(&theta;,&phi;)</td>" in content, content


# @pytest.mark.ci
# @pytest.mark.parametrize("qspiral", [(0, 0, 0), [0.5, 0., 0.],
#                                      array([0.11, 0.11, 0.11])])
# def test_spinorbit(asr_tmpdir_w_params, test_material, get_webcontent,
#                    qspiral):
#     """Test of spinorbit recipe."""
#     from asr.spinorbit import calculate, main
#     from asr.gs import calculate as gs_calculate
#     from asr.spinspiral import spinspiral
#     from ase.parallel import world

#     # test_material.write('structure.json')
#     magmoms = test_material.get_initial_magnetic_moments()
#     calculator = {"mode": {"name": "pw", "ecut": 300},
#                   "kpts": {"density": 6, "gamma": True},
#                   "txt": "gs.txt"}

#     projected=False
#     if qspiral is not None:
#         calculator['mode']['qspiral'] = qspiral
#         projected=True

#     spinspiral(calculator=calculator, atoms=test_material)
#     calculate(gpw='gs.gpw', distance=100.0, projected=projected, width=0.001)
#     result = main()

#     if isinstance(qspiral, ndarray) and all(qspiral == [0.11, 0.11, 0.11]) \
#        or qspiral == [0.5, 0., 0.]:
#         assert result.projected_soc is True
#     elif qspiral is None or qspiral == [0., 0., 0.] or qspiral == (0, 0, 0):
#         assert result.projected_soc is False
#     else:
#         assert False, f'Case was not tested: qspiral = {qspiral}'

#     if world.size == 1:
#         content = get_webcontent()
#         assert "<td>Spinorbitbandwidth(meV)</td>" in content, content
#         assert "<td>SpinorbitMinimum(&theta;,&phi;)</td>" in content, content


@pytest.mark.ci
@pytest.mark.parametrize("distance", [360, 100, 2])
def test_sphere_symmetries(distance):
    from asr.spinorbit import sphere_points

    def to_xyz(theta, phi):
        """ Converts theta, phi spherical coordinates to cartesian coordinates

        Input angles in degrees
        """
        from numpy import pi, sin, cos
        theta *= pi / 180
        phi *= pi / 180
        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)
        return x, y, z

    theta, phi = sphere_points(distance=distance)
    x, y, z = to_xyz(theta, phi)
    assert sum(x) < 1e-8, f"The x-direction seem unsymmetric, {sum(x)} < 1e-8"
    assert sum(y) < 1e-8, f"The y-direction seem unsymmetric, {sum(y)} < 1e-8"
