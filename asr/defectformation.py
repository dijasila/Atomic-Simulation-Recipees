"""Defect formation energies."""
from typing import Union
from asr.core import command, option, ASRResult


group = 'property'
creates = []  # what files are created


@command('asr.defectformation',
         resources='1:2h',
         dependencies=['asr.setup.defects', 'asr.gs'],
         creates=None)
# later include 'asr.dielectricconstant' and 'asr.relax')
# and also the files that this recipe creates
@option('--pristine', type=str,
        help='Name of the groundstate .gpw file of the pristine system. It '
             'always has to be somewhere within a folder that is called '
             '"pristine" in order to work correctly.')
@option('--defect', type=str,
        help='Name of the groundstate .gpw file of the defect systems. They '
             'always have to be within a folder for the specific defect with '
             'a subfolder calles "charge_q" for the respective chargestate q '
             'in order to work correctly.')
@option('--defect_name', type=str,
        help='Runs recipe for all defect folder within your directory when '
             'set to None. Set this option to the name of a desired defect '
             'folder in order for it to run only for this particular defect.')
def main(pristine: str = 'gs.gpw', defect: str = 'gs.gpw',
         defect_name: Union[str, None] = None) -> ASRResult:
    """
    Calculate formation energy of defects.

    This recipe needs the directory structure that was created with
    setup.defects in order to run properly. It has to be launched within the
    folder of the initial structure, i.e. the folder where setup.defects was
    also executed.
    """
    from ase.io import read
    from asr.core import write_json, read_json
    from gpaw.defects import ElectrostaticCorrections
    from pathlib import Path
    import numpy as np
    q, epsilons, path_gs = check_and_get_general_inputs()
    atoms = read('unrelaxed.json')
    nd = int(np.sum(atoms.get_pbc()))
    gs_dict = read_json('results-asr.gs.json')
    defectformation_dict = {}
    defectformation_dict['gaps_nosoc'] = gs_dict.get('gaps_nosoc')
    defectformation_dict['gaps_soc'] = gs_dict.get('gaps_soc')

    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if nd == 3:
        epsilon = (epsilons[0] + epsilons[1] + epsilons[2]) / 3.
        dim = '3d'
    elif nd == 2:
        epsilon = [(epsilons[0] + epsilons[1]) / 2., epsilons[2]]
        dim = '2d'

    folder_list = []
    p = Path('.')

    # Either run for all defect folders or just a specific one
    if defect_name is None:
        [folder_list.append(x) for x in p.iterdir() if x.is_dir()
            and not x.name == 'pristine_sc']
    else:
        [folder_list.append(x) for x in p.iterdir() if x.is_dir()
            and x.name == defect_name]

    for folder in folder_list:
        s = Path(folder.name)
        sub_folder_list = []
        [sub_folder_list.append(x) for x in s.iterdir() if x.is_dir()]
        e_form = []
        # e_fermi = []
        charges = []
        # e_fermi_calc = []
        for sub_folder in sub_folder_list:
            sub_folder_path = folder.name + '/' + sub_folder.name
            setup_params = read_json(sub_folder_path + '/params.json')
            chargestate = setup_params.get('charge')
            charged_file = find_file_in_folder('gs.gpw',
                                               sub_folder_path)
            elc = ElectrostaticCorrections(pristine=path_gs,
                                           charged=charged_file,
                                           q=chargestate, sigma=sigma,
                                           dimensionality=dim)
            elc.set_epsilons(epsilon)
            if chargestate == 0:
                e_form.append(elc.calculate_uncorrected_formation_energy())
            else:
                e_form.append(elc.calculate_corrected_formation_energy())
            charges.append(chargestate)
            # calc = GPAW(find_file_in_folder('gs.gpw', sub_folder_path))
            # e_fermi_calc.append(calc.get_fermi_level())
            # e_fermi.append(0)
        defectformation_dict[folder.name] = {'formation_energies': e_form,
                                             'chargestates': charges}
        # 'fermi_energies_c': e_fermi_calc
        # 'fermi_energies': e_fermi,
    write_json('defectformation.json', defectformation_dict)

    return None


def check_and_get_general_inputs():
    """Determine whether all necessary files exist.

    Checks if all necessary input files and input parameters for
    this recipe are acessible.
    """
    from asr.core import read_json

    # first, get path of 'gs.gpw' file of pristine_sc, as well as the path of
    # 'dielectricconstant.json' of the pristine system
    path_epsilon = find_file_in_folder('dielectricconstant.json', None)
    path_gs = find_file_in_folder('gs.gpw', 'pristine_sc')
    path_q = find_file_in_folder('general_parameters.json', None)

    # if paths were found correctly, extract epsilon and q
    gen_params = read_json(path_q)
    params_eps = read_json(path_epsilon)
    q = gen_params.get('chargestates')
    epsilons = params_eps.get('local_field')
    if q is not None and epsilons is not None:
        msg = 'INFO: number of chargestates and dielectric constant '
        msg += 'extracted: q = {}, eps = {}'.format(q, epsilons)
        print(msg)
    else:
        msg = 'either number of chargestates and/or dielectric '
        msg += 'constant of the host material could not be extracted'
        raise ValueError(msg)

    if path_gs is not None:
        print('INFO: check of general inputs successful')

    return q, epsilons, path_gs


def find_file_in_folder(filename, foldername):
    """Find a specific file in folder.

    Finds a specific file within a folder starting from your current
    position in the directory tree.
    """
    from pathlib import Path

    p = Path('.')
    find_success = False

    # check in current folder directly if no folder specified
    if foldername is None:
        check_empty = True
        tmp_list = list(p.glob(filename))
        if len(tmp_list) == 1:
            file_path = tmp_list[0]
            print('INFO: found {0}: {1}'.format(filename,
                                                file_path.absolute()))
            find_success = True
        else:
            print('ERROR: no unique {} found in this directory'.format(
                filename))
    else:
        tmp_list = list(p.glob('**/' + foldername))
        check_empty = False
    # check sub_folders
    if len(tmp_list) == 1 and not check_empty:
        file_list = list(p.glob(foldername + '/**/' + filename))
        if len(file_list) == 1:
            file_path = file_list[0]
            print('INFO: found {0} in {1}: {2}'.format(
                filename, foldername, file_path.absolute()))
            find_success = True
        elif len(file_list) == 0:
            print('ERROR: no {} found in this directory'.format(
                filename))
        else:
            print('ERROR: several {0} files in directory tree: {1}'.format(
                filename, tmp_list[0].absolute()))
    elif len(tmp_list) == 0 and not check_empty:
        print('ERROR: no {0} found in this directory tree'.format(
            foldername))
    elif not check_empty:
        print('ERROR: several {0} folders in directory tree: {1}'.format(
            foldername, p.absolute()))

    if not find_success:
        file_path = None

    return str(file_path.absolute())


def collect_data():
    # from ase.io import jsonio
    # from pathlib import Path
    # if not Path('results_defectformation.json').is_file():
    #     return {}, {}, {}

    # kvp = {}
    # data = {}
    # key_descriptions = {}
    # dct = jsonio.decode(Path('results_defectformation.json').read_text())

    return None


# ========================================================================== #
# This function doesn't exist in the new asr version anymore and has to be   #
# changed at a later stage, as well as the way the plots are created (db)    #
# ========================================================================== #
# def postprocessing():
#     from asr.utils import read_json
#
#     formation_dict = read_json('defectformation.json')
#     transitions_dict = {}
#     gap = formation_dict.get('gaps_nosoc').get('gap')
#     for element in formation_dict:
#         if element != 'gaps_soc' and element != 'gaps_nosoc':
#             print(element)
#             plotname = element
#             defect_dict = formation_dict[element]
#             transitions_dict[plotname] = plot_formation_and_transitions(
#                 defect_dict, plotname, gap)
#
#     return transitions_dict


def line_intersection(line1, line2):
    """Get intersection between two lines.

    Helper function to calculate intersection of two given lines.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def line(p1, p2):
    """Define a line between p1 and p2.

    Helper function to define a line.
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    """Get intersection between two lines L1 and L2.

    Helper function to calculate intersection of two given lines that were
    defined with the upper 'line' function.
    """
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def plot_formation_and_transitions(defect_dict, defectname, gap):
    """XXX: Do me.

    Function to plot formation energies versus the Fermi energy and to
    obtain transition points between most stable charge states of a given
    defect
    """
    # from asr.utils import write_json
    import matplotlib.pyplot as plt
    import numpy as np

    # initalize and read formation energies and charge states
    x = []
    y = []
    q = []
    x = np.array(defect_dict['fermi_energies'])
    y = np.array(defect_dict['formation_energies'])
    q = np.array(defect_dict['chargestates'])

    # set general parameters
    x_range = np.array([0, gap])
    x_diff = np.array([[-x[0], x_range[1] - x[0]]])
    y_edges = np.array(
        [[y[0] + x_diff[0][0] * q[0], y[0] + x_diff[0][1] * q[0]]])

    # set general plotting parameters
    plt.figure()
    lw = 1
    linestylelist = ['solid', 'dashdot', 'dashed', 'dotted']
    colorlist = ['black', 'C0', 'C1']
    # plt.ylim(0, max(y_edges[:, 0]) * 1.1)
    plt.ylim(0, 20)
    plt.xlim(x_range[0] - 0.1 * gap, x_range[1] + 0.1 * gap)
    # bbox = {'fc': '0.8', 'pad': 0}

    # initialise np array containing all lines
    linearray = np.array([[line([x_range[0], y_edges[0][0]],
                                [x_range[1], y_edges[0][1]]), q[0]]])

    # initialise plot
    plt.plot(x_range, y_edges[0], color=colorlist[np.sign(q[0])], lw=lw,
             linestyle=linestylelist[abs(q[0])], label='q = {}'.format(q[0]))
    plt.text((2 * x_range[0] - 0.1 * gap) / 2.,
             max(y_edges[:, 0]) * 1.1 / 2., 'valence band',
             {'ha': 'center', 'va': 'center'}, rotation=90)
    plt.text((2 * x_range[1] + 0.1 * gap) / 2.,
             max(y_edges[:, 0]) * 1.1 / 2., 'conduction band',
             {'ha': 'center', 'va': 'center'}, rotation=90)

    # append other lines in a loop
    for i in range(1, len(q)):
        x_diff = np.append(x_diff, [[-x[i], x_range[1] - x[i]]], axis=0)
        y_edges = np.append(y_edges, [[y[i] + x_diff[i][0] * q[i],
                                       y[i] + x_diff[i][1] * q[i]]], axis=0)
        linearray = np.append(linearray, [[line([x_range[0], y_edges[i][0]], [
                              x_range[1], y_edges[i][1]]), q[i]]], axis=0)
        plt.plot(x_range,
                 y_edges[i],
                 color=colorlist[np.sign(q[i])],
                 lw=lw,
                 linestyle=linestylelist[abs(q[i])],
                 label='q = {}'.format(q[i]))

    # flip arrays in order for them to start with positive slope
    linearray_up = np.flip(linearray, axis=0)
    x_diff = np.flip(x_diff, axis=0)
    y_edges = np.flip(y_edges, axis=0)
    q_copy = np.flip(q)

    # IMPORTANT: all of the important arrays have now been reversed!
    # find minimum line at zero energy and create temporary array with lines
    for i in range(len(y_edges)):
        if y_edges[i][0] == min(y_edges[:, 0]):
            # start_index = i
            linearray_up = np.delete(linearray_up, np.s_[0:i], 0)
            trans_array = np.array(
                [[(0, y_edges[i][0]), q_copy[i], q_copy[i]]])
            q_copy = np.delete(q_copy, np.s_[0:i])

    # loop over all lines in linearray_up and calculate intersection points
    while len(linearray_up) > 1:
        linedists = np.array([[intersection(linearray_up[0][0],
                                            linearray_up[1][0]),
                              q_copy[0], q_copy[1]]])
        if len(linearray_up) > 2:
            for j in range(2, len(linearray_up)):
                linedists = np.append(linedists, [[intersection(
                    linearray_up[0][0], linearray_up[j][0]), q_copy[0],
                    q_copy[j]]], axis=0)
        linearray_up = np.delete(linearray_up, 0, 0)
        q_copy = np.delete(q_copy, 0)
        x_list = []
        n = 0
        for element in linedists:
            x_list.append(element[0][0])
            if element[0][0] >= 0 and len(x_list) == 1:
                if element[0][0] == min(x_list):
                    trans_array = np.append(trans_array, [element], axis=0)
                    dropout = n
            elif (element[0][0] >= 0 and element[0][0] == min(x_list)
                  and element[0][0] < trans_array[-1][0][0]):
                trans_array = np.append(trans_array, [element], axis=0)
                trans_array = np.delete(trans_array, -2, 0)
                dropout = n
            n = n + 1
        for i in range(dropout):
            linearray_up = np.delete(linearray_up, np.s_[0:i], 0)
    for i in range(len(y_edges)):
        if y_edges[i][1] == min(y_edges[:, 1]):
            trans_array = np.append(trans_array, [[(x_range[1], y_edges[i][1]),
                                                   trans_array[-1][2],
                                                   trans_array[-1][2]]],
                                    axis=0)

    # plot the results and save the figure
    for element in trans_array:
        ratio = element[0][1] / max(y_edges[:, 0])
        plt.axvline(x=element[0][0], ymax=ratio, color='C3',
                    linestyle=(0, (5, 10)))
    plt.axvspan(x_range[0] - 0.1 * gap, x_range[0], color='lightgrey')
    plt.axvspan(x_range[1] + 0.1 * gap, x_range[1], color='lightgrey')
    x_val = [x[0] for x in trans_array[:, 0]]
    y_val = [x[1] for x in trans_array[:, 0]]
    # plt.plot(x_val, y_val, marker='D', linestyle='-', color='C3')
    plt.plot(x_val, y_val, marker='D', linestyle='', color='C3')
    plt.xlabel(r'$E_{F}$ in eV')
    plt.ylabel(r'$E_{formation}$ in eV')
    plotname = 'plot_{}.png'.format(defectname)
    plt.legend()
    plt.savefig(plotname)

    return trans_array


# def webpanel(result, row, key_descriptions):
#    from asr.database.browser import fig, table
#
#    if 'something' not in row.data:
#        return None, []
#
#    table1 = table(row,
#                   'Property',
#                   ['something'],
#                   kd=key_descriptions)
#    panel = ('Title',
#             [[fig('something.png'), table1]])
#    things = [(create_plot, ['something.png'])]
#    return panel, things


# def create_plot(row, fname):
#    import matplotlib.pyplot as plt
#
#    data = row.data.something
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(data.things)
#    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
