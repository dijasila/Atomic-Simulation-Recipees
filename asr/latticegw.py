def latticegw(planewavecutoff=10):
    
    ## --- Imports --- ##
    ## -- maths
    import numpy as np
    from ase.units import Hartree
    ## -- plotting
    import matplotlib 
    #matplotlib.use('tkagg') #REMOVE WHEN RUNNING IN NIFLHEIM
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    ## -- data
    import json                 # java-script object notation (easy for humans to read/write and for machines to parse/generate)
    from ase.io import jsonio
    import os
    from os.path import exists
    from ase.parallel import paropen
    import pickle
    ## -- DFT
    from ase.dft.kpoints import monkhorst_pack
    ## -- GPAW
    from gpaw import GPAW, FermiDirac
    from gpaw.response.df import DielectricFunction
    from gpaw.response.pair import PairDensity
    from gpaw.wavefunctions.pw import PW
    from gpaw.wavefunctions.pw import PWDescriptor
    from gpaw.mpi import world
    from gpaw.response.gw_bands import GWBands
    from gpaw.kpt_descriptor import KPointDescriptor
    ## -- c2db
    from c2db.bsinterpol import interpolate_bandlines2
    ## -- recipe
    from rmr.phonons import phonons 
    
    ## --- Generate main directory for output --- ##
    if not exists('GW+lat-bs'):
        os.makedirs('GW+lat-bs')
    
    #if not exists(f'results_qecut[eV]={qecut*Hartree}'):	# create q-ecut off specific result file
    #    os.makedirs(f'results_qecut[eV]={qecut*Hartree}')
    
    #if not exists(f'eta[eV]={eta*Hartree}'):                	# create eta value specific result file
    #    os.makedirs(f'eta[eV]={eta*Hartree}')

    ## --- Print all output to .txt file --- ##
    #import sys
    #orig_stdout = sys.stdout
    #f = open('GW+lat-bs/gwlat_out.txt','w')
    #sys.stdout = f

    print('')
    print('Calculating GW lattice contribution')
    print('')

    ## --- Groundstate DFT --- ##
    # We will assume the groundstate calculation has been made and simply load the .gpw file in the current directory
    print('Loading groundstate calculation')
    calc = GPAW('gs.gpw', txt=None)             # Define calculator based on groundstate calculation
    print('     Done.')

    ## --- Phonon calculation --- ##
    # Vibrational frequency (phonon) mode
    # Again assuming a phonon calculation has been made and can be loaded in here
    print('Loading phonons')
    p = phonons()						# find phonon data in the current directory
    path = [[0, 0, 0]]						# gamma point calculation for q -> 0 limit
    p.read()							# reads dynamical matrix for eigenmodes
    omega_kl, u_klav = p.band_structure(path, modes=True)	# using the ASE.band_structure function to load frequencies and displacement modes (in eV and Ang/sqrt(amu))
    u_klav *= 1 / np.sqrt(1822.88)				# convert from atomic mass unit (amu) to electron mass

    print('     Done.')

    ## --- Born Charge Calculation --- ##
    # Born charges are also pre-calculated and will be loaded in
    print('Extracting effective born charges')
    # The aim is to extract the 'effective born charges' which are the dot products between borns charges and outer products of the dynamic matrix eigenstates
    os.chdir('./data-borncharges/')         	# change to borncharge data directory (MAYBE THIS CAN BE MADE INTO AN INPUT ARGUMENT)
    filename = 'borncharges-0.01.json'          # load born charges - 0.01 Å displacement (small and sufficient for these calculations)

    with paropen(filename, 'r') as fd:
        data = jsonio.decode(json.load(fd))     # prepares the .json file for data extraction
    Z_avv = data['Z_avv']                       # indices: a-atom, v-(x,y,z), v-(x,y,z). Born charge loaded from .json file
    u_lav = u_klav[0]                           # indices: l-mode, a-atom, v-(x,y,z). Displacement modes read from phonon calculation 
    nmodes, natoms, nv = u_lav.shape
    u_lx = u_lav.reshape(-1, natoms * 3)        # combines the a-atom and v-cartesian coordinates into a single index
    Z_xv = Z_avv.reshape(-1, 3)                 # same as above

    Z_lv = np.dot(u_lx, Z_xv)                   # dot product between displacement mode and born charge THIS IS THE EFFECTIVE BORN CHARGE
    Z2_lvv = []                                 # Z-squared
    for Z_v in Z_lv:
        Z2_vv = np.outer(Z_v, Z_v)              # outerproduct is taken for each l-mode for Z^2 vector
        Z2_lvv.append(Z2_vv)
    
    ind = np.argmax(np.abs(Z2_lvv))             # Index of largest contributing born charge mode
    ind = np.unravel_index(ind, (nmodes, 3, 3)) 
    mode = ind[0]                               # assign mode from index above
    Z2_vv = Z2_lvv[mode]                        # finally the effective born charge squared of this mode
    print('     Done.')

    os.chdir('../')                             # go back to main directory

    ## -- New groundstate calculation --- ##
    if not exists('GW+lat-bs/gwlatgs.gpw'):
        print('Performing new groundstate calculation')
        calc = GPAW('gs.gpw',
                    fixdensity=True,            	# Use same density -> effective potential as in previous calculation
                    kpts={'density': 6.0, 
                          'even': True,
                          'gamma': True},       	# Increase k-point sampling for more accurate Kohn-Sham Eigen-energies and pair densities ???
                    nbands=-10,                 	# additional 10 empty bands
                    symmetry='off',
                    convergence={'bands': -5},  	# converge all bands except last 5
                    txt='GW+lat-bs/gwlatgs.txt')        # .txt file output    

        calc.get_potential_energy()
        # calc.diagonalize_full_hamiltonian()
        calc.write('GW+lat-bs/gwlatgs.gpw', mode='all')      # new .gpw file
        print('     Done.')
    else:
        calc = GPAW('GW+lat-bs/gwlatgs.gpw', txt=None)

    ## --- Electronic dielectric constant in infrared --- ##
    print('Calculating infrared regime electronic dielectric constant (epsilon_inf)')
    df = DielectricFunction('G0W0/gwgs.gpw',txt='eps_inf.txt',name='chi0')   		# Calculator for dielectric function
    epsmac = df.get_macroscopic_dielectric_constant()[1]                		# infrared regime electronic dielectric - eps_inf [1]-with local field effects
    print('     Done.')

    ## --- Pair-densities --- ##   
    print('Preparing for calculation of pair-densities')
    # The aim here is to calculate the pair-densities
    ecut = 10
    pair = PairDensity('GW+lat-bs/gwlatgs.gpw', ecut, response='density')	# Set up calculation for pair-density (default response = 'density')
    nocc = pair.nocc1								# number of occupied bands
    print('     Done.')

    ## --- Calculation loop for n-bands and k-points --- ## 
    # Prepare loop - this loop will calculate the lattice correction term for all n-band and k-points. Stored in a matrix of n-rows and k-columns.
    # The loop will calculate k-point pairs, pair densities, and Kohn-Sham transition energies in each iteration.
    kpts = calc.wfs.kd.bzk_kc                                		# k-points
    ikpts = calc.wfs.kd.ibzk_kc                              		# irreducible k-points (due to symmetries there are equivalent k-points)
    nikpts = len(ikpts)                                      		# number of irreducible k-points
    N_c = calc.wfs.kd.N_c

    # various variables (bands and spin)
    s = 0                                                    		# spin
    nall = nocc+5                                            		# all bands occupied +5 unoccupied
    n_n = np.arange(0,nall)                                  		# n-bands (nall - nocc occupied bands and 5 unoccupied bands)
    m1 = 0          
    m2 = nocc
    m_m = np.arange(m1, m2)                                  		# m-bands (occupied bands only)
    
    # constants for lattice correction term 
    volume = pair.vol                                       		# unit cell volume
    Z = np.sqrt(Z2_vv[0, 0] / volume)                        		# effective born charge at gamma point ???
    eta = 0.001 / Hartree					        # infitesimal eta
    eps = epsmac                                            		# infrared frequency electronic dielectric constant (epsilon_inf) 
    ZBM = ( volume*eps*(0.00299**2 - 0.00139**2)/(4 * np.pi) )**(1/2)
    prefactor = (((4 * np.pi * ZBM) / eps)**2 * 1/volume)      		# derived prefactor based on Born charge
    prefactorBM = 4 * np.pi * (0.00299**2 - 0.00139**2)/eps     	# Botti and Marques prefactor based on w_LO and w_TO
    freqTO = omega_kl[0, mode] / Hartree                       		# phonon frequency mode - converted to Hartree
    freqLO = ( freqTO**2 + 4 * np.pi * ZBM**2 / (eps*volume) )**(1/2)
    
    print('Cell volume:', volume)
    print('IR dielectic:', eps)
    print('omegaTO:', freqTO)
    print('omegaLO:', freqLO)
    print('Z:', Z)
    print('ZBM:', ZBM)
    print('prefactor:', prefactor)
    print('prefactorBM:', prefactorBM)
    
    sigmalat_temp_nk = np.zeros([nall,nikpts],dtype=complex) 		# empty complex array for lattice correction terms in n-bands and k-points
    
    print('Begin loop over n-bands and k-points to calculate lattice correction matrix')
    print('    Calculates k-point pairs,')
    print('               pair-densities,')
    print('               Kohn-Sham transition energies')
    print('    for each k and n')
    print('Lattice correction values are stored in an n-by-k matrix')

    ## --- q-dependency
    offset_c = 0.5 * ((N_c + 1) % 2) / N_c
    bzq_qc = monkhorst_pack(N_c) + offset_c
    bzq_qv = np.dot(bzq_qc, calc.wfs.gd.icell_cv) * 2 * np.pi

    qabs_q = np.sum(bzq_qv**2,axis=1)
    #qecut = 20.0 / Hartree						# energy cut off for q points in Hartree
    qecut = 0.5 / Hartree
    #qecut = 0.01 / Hartree
    #qecut = 0.1 / Hartree
    mask_q = qabs_q / 2 < qecut
    bzq_qc = bzq_qc[mask_q]
    nqtot = len(mask_q)
    nq = len(bzq_qc)
 
    mybzq_qc = bzq_qc[world.rank::world.size]
    myiqs = np.arange(nq)[world.rank::world.size]
    
    print('total number of q:', nqtot)
    print('number of included q:', nq)
    
    prefactor *= nqtot/(2*np.pi)**3 * volume				# micro cylinder integral prefactor 
    
    print('MicroCylinder Prefactor:', nqtot/(2*np.pi)**3 * volume) 
    
    os.chdir('GW+lat-bs') 
    if not exists(f'results_qecut[eV]={qecut*Hartree}'):		# create q-ecut off specific result file
        os.makedirs(f'results_qecut[eV]={qecut*Hartree}')
    os.chdir(f'results_qecut[eV]={qecut*Hartree}')
    if not exists(f'eta[eV]={eta*Hartree}'):                		# create eta value specific result file
        os.makedirs(f'eta[eV]={eta*Hartree}')
    os.chdir('../../')

    # -- empty matrices to check q-dependence
    corr_nkq = np.zeros([nall,nikpts,nq],dtype=complex)			# empty complex array for q dependance of correction term
    pairdensity_nkq = np.zeros([nall, nikpts, nq])
    transition_nkq = np.zeros([nall, nikpts, nq]) 
    denominator_nkq = np.zeros([nall, nikpts, nq],dtype=complex)
    pole_nkq = np.zeros([nall, nikpts, nq],dtype=complex)
    qabs_qecut = np.zeros(nq)

    for iq, q_c in zip(myiqs, mybzq_qc): 
        print('rank:', world.rank,'iq:', iq, 'q_c:', q_c)
        
        dq1_c = 1 / N_c * [1, 0, 0]
        dq2_c = 1 / N_c * [0, 1, 0]
        dq3_c = 1 / N_c * [0, 0, 1]
        dq_cc = np.array([dq1_c, dq2_c, dq3_c])
        qd = KPointDescriptor([q_c])
        qd1 = KPointDescriptor([q_c + dq1_c])
        qd2 = KPointDescriptor([q_c + dq2_c])
        qd3 = KPointDescriptor([q_c + dq3_c])
        pd = PWDescriptor(ecut, calc.wfs.gd, complex, qd)
        pd1 = PWDescriptor(ecut, calc.wfs.gd, complex, qd1)
        pd2 = PWDescriptor(ecut, calc.wfs.gd, complex, qd2)
        pd3 = PWDescriptor(ecut, calc.wfs.gd, complex, qd3)

        Q_aGii = pair.initialize_paw_corrections(pd)
        q_v = np.dot(q_c, pd.gd.icell_cv) * 2 * np.pi
        q2abs = np.sum(q_v**2)
        B_cv = pd.gd.icell_cv * 2 * np.pi
        E_cv = B_cv / ((np.sum(B_cv**2,1))**(1/2))[:, None]

        qabs_qecut[iq] = q2abs**(1/2)
        
        for k in np.arange(0,nikpts):                               	# loop over k-points
            # set k-point value
            k_c = ikpts[k]
            k_v = np.dot(k_c, pd.gd.icell_cv) * 2 * np.pi		# cartesian coordinates
            # -- K-point pair (k+-q) -- #
            kptpair = pair.get_kpoint_pair(pd, s, k_c, 0, nall, m1, m2)
            kptpair1 = pair.get_kpoint_pair(pd1, s, k_c, 0, nall, m1, m2)
            kptpair2 = pair.get_kpoint_pair(pd2, s, k_c, 0, nall, m1, m2)
            kptpair3 = pair.get_kpoint_pair(pd3, s, k_c, 0, nall, m1, m2)
            # -- Kohn-Sham energy difference (e_n - e_m) -- #
            deps_nm = kptpair.get_transition_energies(n_n, m_m)     	# at band n transition to band m
            deps1_nm = kptpair1.get_transition_energies(n_n, m_m)     	# at band n transition to band m
            deps2_nm = kptpair2.get_transition_energies(n_n, m_m)     	# at band n transition to band m
            deps3_nm = kptpair3.get_transition_energies(n_n, m_m)     	# at band n transition to band m
            
            v_nmc = (np.array([deps1_nm, deps2_nm, deps3_nm]) - deps_nm).transpose(1, 2, 0) / ((np.dot(dq_cc, B_cv)**2).sum(1)**0.5)
            v_nmv = np.dot(v_nmc, np.linalg.inv(E_cv).T)

            # -- Locate k = 0 index
            if np.allclose(k_c, 0):
                k0 = k                    
            # -- Pair-Densities -- #
            pairrho_nmG = pair.get_pair_density(pd, kptpair, n_n, m_m, optical_limit=False, intraband=False, Q_aGii=Q_aGii, extend_head=True)
            if np.allclose(q_c, 0.0):										# in q -> 0 limit
                pairrho2_nm = np.sum(np.abs(pairrho_nmG[:, :, 0:3])**2, axis=-1) / (3*volume * nqtot)		# n /= m terms converted to Hartree
                pairrho2_nm[m_m, m_m] = (1 / (4 * np.pi**2)) * ((48*np.pi**2) / (volume * nqtot))**(1/3)	# n = m terms converted to Hartree
            else:
                pairrho2_nm = (np.abs(pairrho_nmG[:, :, 0])**2 / (volume * nqtot * q2abs))			# finite q terms

            for n in n_n:                                               # loop over bands (occupied and unoccupied)    
                pairrho2_m = pairrho2_nm[n]
                deps_m = deps_nm[n]
                
                # -- Remove degenerate states -- #
                if n < nocc and np.allclose(q_c, 0.0):			# only occupied bands are involved in some and we are interested in the q->0 limit
                    m = np.where(np.isclose(deps_m,0.0))[0]		# index for zero transition energy (transition between degenerate states)
                    for i in np.arange(0,len(m)):		
                         if n != m[i]:					# check if the transition is infact between degenerate states and not the same state
                            pairrho2_m[m[i]] = 0.0			# remove contribution
                
                # -- Remove singularities -- #
                denom = np.real((deps_m - 1j * eta)**2  - freqLO**2)
                if any(np.isclose(denom,0.0,1e-05,1e-08)):
                #   print('Denom:',denom)
                #   print(np.isclose(denom,0.0,1e-05,1e-08))
                   indx = np.where(np.isclose(denom,0.0,1e-05,1e-08))[0]
                #   print('index:',indx)
                   pairrho2_m[indx] = 0.0
                
                # -- Lattice correction term -- #
                # micro-cyclinder solution to singularities
                v_mv = v_nmv[n]
                v_m = np.linalg.norm(v_mv, axis=1)						# group velocity
                dq_cv = np.dot(dq_cc, B_cv)							# vector to nearest q
                qz = np.linalg.norm(dq_cv, axis=1).sum() /3
                #qz_m = (1/2) * np.linalg.norm(np.dot(dq_cv, v_mv.T / v_m), axis=0)		# half cylinder length
                qrvol = abs(np.linalg.det(dq_cv))						# parallelapiped volume of q-space
                qr = np.sqrt(qrvol / (qz * np.pi))						# cylinder radius based on above volume
                
                muVol_m = np.pi * qr**2 / (freqLO * v_m) * (np.arctanh((deps_m - v_m * qz - 1j*eta) / freqLO) - np.arctanh((deps_m - 1j*eta) / freqLO))  
                
                corr = np.sum(pairrho2_m * muVol_m)										# micro-volume correction
                
                #print('microvol:', muVol_m)
                #print('standard:', qrvol / ((deps_m - 1j * eta)**2  - freqTO**2 - ((4 * np.pi * ZBM**2) / (volume*eps)) ))

                #corr = np.sum(pairrho2_m / ((deps_m - 1j * eta)**2  - freqTO**2 - ((4 * np.pi * ZBM**2) / (volume*eps)) ))	# derived correction
                #corr = np.sum(pairrho2_m / ((deps_m - 1j * eta)**2  - 0.00299**2)) 						# Botti and Marques correction
                
                sigmalat_temp_nk[n,k] += corr
                
                # --  q-dependence -- # 
                corr_nkq[n,k,iq] = corr
                pairdensity_nkq[n,k,iq] = np.sum(pairrho2_m)
                transition_nkq[n,k,iq] = deps_m[n] if n < nocc else deps_m[nocc-1]
                denominator_nkq[n,k,iq] = (transition_nkq[n,k,iq] - 1j * eta)**2 - freqTO**2 - (4 * np.pi * ZBM**2) / eps 
                pole_nkq[n,k,iq] = (transition_nkq[n,k,iq] - 1j * eta)**2
                
    world.sum(sigmalat_temp_nk)
    world.sum(corr_nkq)
    world.sum(pairdensity_nkq)
    world.sum(transition_nkq)
    world.sum(denominator_nkq)
    world.sum(pole_nkq)
    world.sum(qabs_qecut)


    ## -- sorting arrays by qabs
    order = np.argsort(qabs_qecut)

    corr_nkq = corr_nkq[:,:,order]
    pairdensity_nkq = pairdensity_nkq[:,:,order]
    transition_nkq = transition_nkq[:,:,order]
    denominator_nkq = denominator_nkq[:,:,order]
    pole_nkq = pole_nkq[:,:,order]
 
    os.chdir(f'GW+lat-bs/results_qecut[eV]={qecut*Hartree}/eta[eV]={eta*Hartree}')	# change result correct result folder generated at start

    sigmalat_nk = prefactor * sigmalat_temp_nk * Hartree				# Lattice correction matrix converted to eV
    
    print('     Done.')
    
    print('rows (n-bands):', sigmalat_nk.shape[0],',', 'columns (k-points):', sigmalat_nk.shape[1])

    if world.rank == 0:
        with open(f'sigma-lattice_nk_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
            json.dump(jsonio.encode(sigmalat_nk), outfile)
   
    ## --- q-dependancies --- ##
    if nq > 1:
        # -- correction term
        corr_nq = np.mean(corr_nkq.real,axis=1) * prefactor.real * Hartree
        corr_nk0q = corr_nkq[:,k0,:].real * prefactor.real * Hartree
        
        if world.rank == 0:
            with open(f'corr_nq_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(corr_nq), outfile)
        if world.rank == 0:
            with open(f'corr_nk0q_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(corr_nk0q), outfile)
        
        # -- pair densities
        pairdensity_nq = np.mean(pairdensity_nkq,axis=1)
        pairdensity_nk0q = pairdensity_nkq[:,k0,:]

        if world.rank == 0:
            with open(f'pairdensity_nq_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(pairdensity_nq), outfile)
        if world.rank == 0:
            with open(f'pairdensity_nk0q_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(pairdensity_nk0q), outfile)
        
        # -- transition energies
        transition_nq = np.mean(transition_nkq,axis=1) * Hartree
        transition_nk0q = transition_nkq[:,k0,:] * Hartree

        if world.rank == 0:
            with open(f'transition_nq_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(transition_nq), outfile)
        if world.rank == 0:
            with open(f'transition_nk0q_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(transition_nk0q), outfile)
        
        # -- denominator
        denominator_nq = np.mean(denominator_nkq.real,axis=1) * Hartree
        denominator_nk0q = denominator_nkq[:,k0,:].real * Hartree

        if world.rank == 0:
            with open(f'denominator_nq_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(denominator_nq), outfile)
        if world.rank == 0:
            with open(f'denominator_nk0q_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(denominator_nk0q), outfile)
        
        # -- omegaLO points
        omegaLO2 = (freqLO  * Hartree)**2 
        pole_nq = np.mean(pole_nkq,axis=1).real * Hartree
        pole_nk0q = pole_nkq[:,k0,:].real * Hartree

        if world.rank == 0:
            with open(f'omegaLO2.json','w') as outfile:
                json.dump(jsonio.encode(omegaLO2), outfile)
        if world.rank == 0:
            with open(f'pole_nq_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(pole_nq), outfile)
        if world.rank == 0:
            with open(f'pole_nk0q_{qecut*Hartree}_{eta*Hartree}.json','w') as outfile:
                json.dump(jsonio.encode(pole_nk0q), outfile)
        
    ## --- Combine with GW --- ##
    print('Correcting GW calculation')
    os.chdir('../../../')									    # back to main directory
    
    # -- load GW data
    gwdata = pickle.load(open('G0W0/g0w0_results.pckl', 'rb'), encoding='latin1')     # load results from GW calculation
    
    # -- load in DFT eigen energies
    dft_skn = gwdata['eps']
    
    # -- load in GW eigen energies
    print('Loading GW quasi-particle Eigen-energies')
    #print(gwdata.keys())
    gw_skn = gwdata['qp']                                                                           # extract quasi-particle energies
    gwldata = gwdata.copy()                                                                         # copy this matrix for alterations to avoid altering the original
    gw_kn = gw_skn[0,:,:]                                                                           # no s-spin polarisation

    # -- load in lattice correction
    corr_kn = np.transpose(sigmalat_nk.real)                                            # take real values 
    corr_skn = np.expand_dims(corr_kn, axis=0)
    print('     Done.')

    # -- corrected GW eigen values
    print('Correcting Eigen-energies')
    
    #print('GW_kn -')
    #print('columns (n-bands):', gw_kn.shape[1],',', 'rows (k-points):', gw_kn.shape[0])     # number of k-point and n-bands in GW quasiparticle energy matrix
    #print('GW_k0n:', np.round(gw_kn[k0,:], 3))
    #print('Lat_nk -')
    #print('rows (n-bands):', lat_nk.shape[0],',', 'columns (k-points):', lat_nk.shape[1])   # number of k-point and n-bands in lattice correction matrix
    #print('lat_nk0:', np.round(lat_nk[:, k0], 3))
 
    #corr_kn = gwldata['qp'][0,:,:] - np.transpose(lat_nk[0:7,:])                            # corrected eigen energies, for upper valence band (3) and lower conduction band (4)
    #corr_skn = np.expand_dims(corr_kn, axis=0)
    
    #print('Corr_kn -')
    #print('columns (n-bands):', corr_kn.shape[1],',', 'rows (k-points):', corr_kn.shape[0]) # number of k-point and n-bands in corrected energy matrix
    #print('Corr_k0n:', np.round(corr_kn[k0,:], 3))
    print('     Done.')

    # -- interpolate bands
    print('Interpolating bands along k-path')
    
    gwcalc = GPAW('G0W0/gwgs.gpw', txt=None)    
    path = 'LGX'                                                        # dependent on the reciprocal struct. LiF is FCC so it has a specific path. See ASE documentation for paths

    eDFT_skn = dft_skn[:,:,:]
    DFT_results = interpolate_bandlines2(gwcalc, path, eDFT_skn)
 
    e_skn = gw_skn[:,:,:]                                               # set GW eigenvalues
    GW_results = interpolate_bandlines2(gwcalc, path, e_skn)            # interpolates between k-points for bandstructure

    ecorr_skn = corr_skn[:,:,:]                                         # set corrected eigenvalues
    GWlat_results = interpolate_bandlines2(calc, path, ecorr_skn)       # interpolates between k-points for bandstructure

    if world.rank == 0:
        np.save('GW+lat-bs/egw_skn.npy', e_skn)
        np.save('GW+lat-bs/ecorr_skn.npy', ecorr_skn)
        np.save('GW+lat-bs/edft_skn.npy', eDFT_skn)

    # -- extract data
    xDFT = DFT_results['x']                                             # extract k-points for x-axis
    eDFT_kn = DFT_results['e_skn'][0]
    DFT_gap = np.min(eDFT_kn[:, nocc]) - np.max(eDFT_kn[:, nocc-1]) 

    xGW = GW_results['x']                                               # extract k-points for x-axis
    eGW_kn = GW_results['e_skn'][0]                                     # extract GW eigen energies [0] for no spin
    GW_gap = np.min(eGW_kn[:, nocc]) - np.max(eGW_kn[:, nocc-1])        # GW band gap between lowest conduction band (nocc) and upper valence band (nocc-1) eigen energy
    
    ecorr_kn = GWlat_results['e_skn'][0]                                # extract corrected eigen energies [0] for no spin
    elat_kn = eGW_kn - ecorr_kn[:,0:7]					# CORRECT QP ENERGIES WITH LATTICE CORRECTION
    xlat = GWlat_results['x']                                           # extract k-points for x-axis
    GWlat_gap = np.min(elat_kn[:, nocc]) - np.max(elat_kn[:, nocc-1])   # GW band gap between lowest conduction band (nocc) and upper valence band (nocc-1) eigen energy
    
    X = GW_results['X']                                                 # extract vertices
    labels_K = [r'$L$', r'$\Gamma$', r'$X$']                            # vertex labels
    
    print('DFT direct gap =', DFT_gap, 'eV')
    print('GW direct gap =', GW_gap, 'eV')
    print('GW+lat direct gap =', GWlat_gap, 'eV')
    print('     Done.')

    # -- plot results
    print('Plotting bands and saving into directory: GW+lat-bs')
    os.chdir(f'GW+lat-bs/results_qecut[eV]={qecut*Hartree}/eta[eV]={eta*Hartree}') 
    if world.rank == 0:
       with open('nq.json','w') as outfile:
            json.dump(jsonio.encode(nq), outfile)
    #plot = plt.figure(figsize=(12,5))
    #for eDFT_k in eDFT_kn.T:
    #     plt.plot(xGW, eDFT_k, '-b', linewidth=2)
    #for eGW_k in eGW_kn.T:
    #     plt.plot(xGW, eGW_k, '-k', linewidth=2)
    #for elat_k in elat_kn.T:
    #     plt.plot(xlat, elat_k, '-r', linewidth=2)
    #for p in X:
    #    plt.axvline(p, color='k', linestyle=':', linewidth=1.7)
    #plt.xticks(X, labels_K, fontsize=18)
    #plt.xlim(X[0], X[-1])
    #plt.ylabel('Energy (eV)', fontsize=24)
    #leg_handles = [mpl.lines.Line2D([], [], linestyle='-', marker='', color='b'),
    #               mpl.lines.Line2D([], [], linestyle='-', marker='', color='k'),
    #               mpl.lines.Line2D([], [], linestyle='-', marker='', color='r')]
    #leg_labels = ['DFT', r'G$_0$W$_0$',r'G$_0$W$_0$+Lattice']
    ##plt.legend(leg_handles, leg_labels, loc='right', fontsize=15)
    ##plt.legend()
    #plt.legend(leg_handles, leg_labels, bbox_to_anchor=(1.56,0.86), loc='right', fontsize=17)
    #plt.title(fr'No. of q = {nq}, $\eta = {eta*Hartree}\,eV$')
    #textstr = 'DFT direct gap: %.3f eV\nGW direct gap: %.3f eV\nGW+lat direct gap: %.3f eV\nExperimental gap: 14.20 eV'%(DFT_gap, GW_gap, GWlat_gap)
    #plt.gcf().text(0.62, 0.45, textstr, fontsize=10)
    #plt.tight_layout()
    #if world.rank == 0:
    #    plt.savefig(f'gwlat_bs_{qecut*Hartree}_{eta*Hartree}.pdf')

    #if world.rank == 0:
    #    plt.show()

    print('Done. Complete.')

    #sys.stdout = orig_stdout
    #f.close()

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate GW Lattice contribution')

    # Set variables
    help = 'Atomic displacement when moving atoms in Å'
    parser.add_argument('-ecut', '--planewavecutoff', help=help, default=10,
                        type=float)

    return parser

def main(args=None):
    latticegw(**args)

if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())
    main(args)
