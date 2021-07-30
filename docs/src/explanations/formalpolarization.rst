.. _Modern theory of polarization:

===============================
 Modern theory of Polarization
===============================

.. contents::

This document contains the detailed explanation of the theory of the
Modern theory of polarization.

Summary
=======

The standard definition of the Polarization density is not
well-defined in periodic system, however, in the Modern theory of
Polarization it is demonstrated that another definition exists that
relates to changes in the Polarization density. This allows for a
straight-forward way to calculate derivatives of the Polarization
density, and is therefore useful for calculating the piezoelectric
tensor which describes the change in the polarization density in
response to an applied strain.

Relevant recipes
================

The following recipes are all relying on the Modern theory of Polarization:

- :py:func:`asr.c2db.formalpolarization.main`
- :py:func:`asr.c2db.piezoelectrictensor.main`
- :py:func:`asr.borncharges.main`

Alternative literature
======================

To learn more about this please consider

- Good resource for beginners: `N.A. Spaldin, "A beginners guide to
  the modern theory of polarization", Journal of Solid State Chemistry
  195 (2012) 2–10. <https://doi.org/10.1016/j.jssc.2012.05.010>`_
- Good review paper: `R. Resta, "Macroscopic polarization in
  crystalline dielectrics: the geometric phase approach",
  Rev. Mod. Phys. 66 (1994)
  899–915. <https://doi.org/10.1103/RevModPhys.66.899>`_
- Paper with historical significance: `R.D. King-Smith, D. Vanderbilt,
  "Theory of polarization of crystalline solids", Phys. Rev. B
  47 (1993) R1651–R1654. <https://doi.org/10.1103/PhysRevB.47.1651>`_
- Paper with historical significance: `R. Resta, "Macroscopic Electric
  Polarization as a Geometric Quantum Phase",
  Eur. Phys. Lett. 22 (1993)
  133–138. <https://doi.org/10.1209/0295-5075/22/2/010>`_

Explanation
===========

The polarisation of a dielectric medium represents the dipole per unit
volume. As such it would be natural to define a macroscopic (i.e. unit
cell averaged) polarisation for an infinite periodic bulk system as

.. math::
   :label: P

   \mathbf P = \frac{e}{V_\textrm{cell}}\int_{V_\textrm{cell}} \mathbf r n(\mathbf r) d\mathbf r

where :math:`n` includes the ion point charges as well as the
delocalised electron charge (:math:`e > 0`). It is, however, obvious
that this definition is not meaningful as it depends on the chosen unit
cell (see p. 365 in Grosso and Parravicini 2nd. ed.). Instead, it turns
out that only *changes* in polarisation are physically meaningful, and
in fact all experimental measurements of bulk polarisation indeed probe
the difference in polarisation between two states of the crystal. In
this and the next section we show, following two different routes, that
such quantities are indeed well defined.

Before we derive a unit-cell independent formula for the change in
polarization, it should be noted that expression :eq:`P` cannot be
directly applied to determine such a change. This point should be clear
from Figure :numref:`Pcell`.

.. figure:: born_charge.png
   :name: Pcell

   An illustration of the dependence of the polarization change on the
   choice of unit cell when computed using Eq. :eq:`P`. Here we induce
   a change in the electron density by moving an atom which could induce
   a change in the polarization of the material. However, depending on
   the choice of unit cell (top and bottom panel) some fractional amount
   of electrons will be folded back into the unit cell
   (:math:`\delta n`) and yield arbitrary values for the induced
   polarisation.

Rather than starting from the unit cell dependent formula :eq:`P`, we
consider the polarization of a finite piece of the bulk for which
:eq:`P` is meaningful when :math:`V_{\textrm{cell}}` is replaced by the
total volume of the crystal, :math:`V`. The idea is now to calculate the
change in :math:`\mathbf P` induced by some change in the Hamiltonian,
and then show that taking the thermodynamic limit (:math:`V\to \infty`)
of the polarisation change is mathematically well defined.

In the following we consider the change in polarisation when the
potential is changed adiabatically from :math:`v_{\lambda=0}` to
:math:`v_{\lambda=1}`. We have

.. math:: \Delta \mathbf P = \int_{0}^{1} \frac{d \mathbf P}{d\lambda}d\lambda

and from Eq. :eq:`P` we can write

.. math::
   :label: dP

   \frac{d \mathbf P}{d\lambda} = -\frac{e}{V} \sum_n^{\text{occ}} \langle \psi_n^\lambda |\mathbf r|\frac{d \psi_n^\lambda}{d \lambda} \rangle + \mathrm{c.c.}

Using first order perturbation theory we have

.. math:: |\frac{d \psi_n^\lambda}{d \lambda} \rangle = \sum_{m\neq n} |\psi_m^\lambda\rangle\frac{\langle \psi^\lambda_m|\frac{\partial v_\lambda}{\partial \lambda}|\psi_n^\lambda\rangle}{\varepsilon_n-\varepsilon_m}.

 Inserting this in ([eq:dP]) we obtain

.. math::

   \frac{d \mathbf P}{d\lambda} = -\frac{e}{V} \sum_n^{\text{occ}}\sum_{m\neq n} \frac{\langle \psi^\lambda_n|\mathbf r |\psi_m^\lambda\rangle
   \langle \psi^\lambda_m|\frac{\partial v_\lambda}{\partial \lambda}|\psi_n^\lambda\rangle}
   {\varepsilon_n-\varepsilon_m} + \mathrm{c.c.}

Using the commutator relation :math:`[\mathbf
r,H_{\lambda}]=i\hbar\mathbf p / m_e`, the off-diagonal matrix
elements of the position operator can be rewritten

.. math:: \langle \psi_n^\lambda |\mathbf r|\psi_m^\lambda  \rangle = \frac{i\hbar}{m_e}\frac{\langle \psi_n^\lambda |\mathbf p|\psi_m^\lambda  \rangle}{\varepsilon_m-\varepsilon_n}

and we finally arrive at the expression

.. math::
   :label: dP_final

   \frac{d \mathbf P}{d\lambda} = \frac{i e \hbar}{Vm_e} \sum_n^{\text{occ}}\sum_{m\neq n} \frac{
   \langle \psi^\lambda_n|\mathbf p |\psi_m^\lambda\rangle \langle \psi^\lambda_m|\frac{\partial v_\lambda}{\partial \lambda}|\psi_n^\lambda\rangle }
   {(\varepsilon_n-\varepsilon_m)^2} + \mathrm{c.c.}

This quantity is well defined for any piece of material also for a
periodic solid in the thermodynamic limit. It does not depend on the
choice of unit cell (because it makes no reference to the unit cell) and
it is independent of the phases chosen for the Bloch states.

Polarization change from Kubo formula
-------------------------------------

In the previous section, the problem with the unit cell dependent
expression :eq:`P`, was circumvented by considering a finite piece of
material and then taking the thermodynamic limit. In this section we
present an alternative formulation which defines the polarization from
the current flowing through a unit cell in response to a periodic
adiabatic change in the potential.

Thus we consider the current flow produced by the adiabatic change in
the potential from :math:`v_{\lambda=0}` to :math:`v_{\lambda=1}`, where
:math:`v_\lambda` is assumed to be periodic for all :math:`\lambda`. The
(microscopic) polarizability is related to the current density via

.. math:: \frac{\partial \mathbf P(\mathbf r) }{ \partial t} =\mathbf j(\mathbf r)

As a quantum mechanical operator we have :math:`\frac{\partial \mathbf
P(\mathbf r) }{ \partial t} = [\mathbf P, H] / i\hbar`. Thus when
considering off-diagonal matrix elements of :math:`\mathbf P` on
energy eigenstates we have

.. math:: \langle \psi_n|\mathbf P(\mathbf r) |\psi_m\rangle = i\hbar\frac{\langle \psi_n|\mathbf j(\mathbf r) |\psi_m\rangle}{\varepsilon_m - \varepsilon_n}.

Since we are interested in the macroscopic polarisation we perform a
unit cell average. Thanks to the Bloch form of the wave functions,
:math:`\psi_{nk}(\mathbf r)=e^{i\mathbf{k}\cdot
\mathbf{r}}u_{nk}(\mathbf r)`, we have

.. math:: \int_{V} \psi_{nk}^* \mathbf j(\mathbf r) \psi_{mk'} d \mathbf r= \frac{e}{m_e} \langle \psi_{nk} |\mathbf p |\psi_{mk}\rangle\delta_{kk'}

Suppose the system is in the ground state of :math:`H(\lambda)`. We
now consider the change in :math:`\mathbf P` when the Hamiltonian is
changed adiabatically to :math:`H(\lambda + d\lambda)`. This change
can be obtained from the Kubo formula using :math:`\mathbf P` as the
observable and :math:`dH(\lambda)=\frac{\partial v_\lambda}{ \partial
\lambda} d\lambda` as the time-independent perturbation. The finite
imaginary frequency :math:`i\eta` in the Kubo formula ensures that the
perturbation is switched on adiabatically so that the system stays in
the ground state.  With this we obtain

.. math::

   \frac{\partial \mathbf P(\lambda)}{\partial \lambda} =
   \frac{-i e \hbar}{Vm_e} \sum_n^{\text{occ}}\sum_{m\neq n}
   \frac{\langle \psi^\lambda_{n}|\frac{\partial v_\lambda}{\partial
   \lambda}|\psi_{m}^\lambda\rangle \langle \psi^\lambda_m|\mathbf p
   |\psi_n^\lambda\rangle} {(\varepsilon_n-\varepsilon_m)^2} +
   \mathrm{c.c.}

which coincide with Eq. :eq:`dP_final`.

Polarisation change as a Berry phase on the occupied manifold
-------------------------------------------------------------

Eq. :eq:`dP\_final` uniquely specifies the macroscopic polarisation
change due to an adiabatic change of the crystal potential. It has the
drawback that it involves a sum over unoccupied states making it costly
to evaluate in practice. As shown below, it is possible to obtain an
expression involving only the occupied subspace. Furthermore, it is
shown that the polarization change, :math:`\Delta \mathbf P`, can be
calculated from knowing only its value at the end points of the
adiabatic path :math:`\lambda=0..1`. This comes, however, at the price
of an introduced ambiguity, namely that the polarisation change can be
determined only up to an integer number of polarisation quanta,
:math:`e L / V_\mathrm{cell}`, where :math:`L` is the unit cell length.
In practice, however, this is not a problem because
:math:`|\Delta \mathbf P|\ll e L / V_\mathrm{cell}`.

We use the relations

.. math::

   \langle \psi^\lambda_{nk}|\frac{\partial v_\lambda}{ \partial \lambda} |\psi^\lambda_{mk}\rangle = \langle u^\lambda_{nk}|[\frac{\partial }{\partial \lambda}, H(\mathbf k,\lambda)]| u^\lambda_{mk}\rangle
   
   \langle \psi^\lambda_{nk}| p_{\alpha} |\psi^\lambda_{mk}\rangle = \frac{m_e}{\hbar}\langle u^\lambda_{nk}|[\frac{\partial }{\partial k_{\alpha}}, H(\mathbf k,\lambda)]| u^\lambda_{mk}\rangle

where the cell periodic Hamiltonian is given by

.. math:: H(\mathbf k,\lambda) = (-i\nabla + \mathbf k)^2 +v_\lambda(\mathbf r).

It should be noted that for the above relations to hold it is essential
that the cell-periodic functions, :math:`u^\lambda_{nk}`, are analytic
with respect to :math:`\mathbf k` and :math:`\lambda`. Substituting into
Eq. :eq:`dP\_final` we obtain (after some manipulations)

.. math:: \Delta P_\alpha = \frac{-e}{(4\pi^3)} \int_{\mathrm{BZ}}d\mathbf k \sum_n^{\text{occ}}\int_0^1 d\lambda\, \mathrm{Im}\left(\langle \frac{\partial u_{nk}^\lambda}{\partial k_\alpha} |\frac{\partial u_{nk}^\lambda}{\partial \lambda} \rangle\right)

XXX (show this!). It can be shown (see e.g. Grosso and Paravicini) that the above
expression can be rephrased as

.. math::
   :label: 1

   \Delta \mathbf P = \Delta \mathbf P_{\mathrm{ion}} + [\mathbf P_{\mathrm{el}}(1)-\mathbf P_{\mathrm{el}}(0)]

where

.. math::
   :label: 2

   \mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3}\mathrm{Im}\int_{\mathrm{BZ}}d\mathbf k \sum_n^{\text{occ}}  \langle u_{nk}^\lambda |\nabla_{\mathbf k}|u_{nk}^\lambda \rangle.

Considering the polarisation along a particular direction, say the
:math:`z`-axis, the derivative only connects Bloch states along
:math:`\mathbf k_z`. In this case the BZ integral can be discretised in
the directions perpendicular to :math:`z`, and the contribution for each
:math:`\mathbf k_{\perp}` becomes

.. math::
   :label: 3

   \mathbf P_{\mathrm{el},z}(\lambda) = \frac{e}{2\pi A}\mathrm{Im}\int_{-\pi/c}^{\pi/c} d k_z \sum_n^{\text{occ}} \langle u_{nk}^\lambda |\frac{\partial u_{nk}^\lambda}{\partial k_z}\rangle

where :math:`A` is the area of the unit cell in :math:`xy` plane. We
can write this as

.. math::

   \mathbf P_{\mathrm{el},z}(\lambda) = \frac{e}{2\pi A}\sum_n^{\text{occ}} \phi_n

where

.. math::

   \phi_n = \mathrm{Im}\int_{-\pi/c}^{\pi/c} d k_z  \langle u_{nk}^\lambda |\frac{\partial u_{nk}^\lambda}{\partial k_z}\rangle

is nothing but the Berry phase picked up along the 1D BZ. As always the
expression is invariant under a change in the phases of the wave
functions, :math:`e^{i\theta(\mathbf k)}`, as long as :math:`\theta` is
differentiable on the BZ torus (i.e. with periodic boundary conditions).
We notice, however, that in contrast to the normal Berry phase, the
Hamiltonian :math:`H(\mathbf k,\lambda)`, from which the cell-periodic
functions derive, is not cyclic over the 1D BZ because
:math:`H(\mathbf k,\lambda)=H(\mathbf k+\mathbf G,\lambda)` only modulus
a gauge transformation, i.e. a unitary transformation of the form
:math:`\exp(i\chi(\mathbf r))`. This means that

.. math::
   :label: periodic

   u_{n\mathbf k}^\lambda = e^{i\mathbf r \cdot \mathbf G}u_{n,\mathbf k+\mathbf G}^\lambda

(which is not just a phase factor). We refer to this relation as the
periodic gauge.

Now, we show that Eqs. (:eq:`1` - :eq:`3`) only determine :math:`\Delta P`
up to an integer number of polarisation quanta. To this end consider the
special case where the Hamiltonians at :math:`\lambda=0` and 1 are
identical, e.g. an atom is moved along a closed loop. In this case
:math:`u_{n\mathbf k}^{(0)}` and :math:`u_{n\mathbf k}^{(1)}` can at
most differ by a phase,

.. math:: u_{n\mathbf k}^{(1)}(\mathbf r) = e^{i\theta_{n\mathbf k}}u_{n\mathbf k}^{(0)}(\mathbf r).

Inserting this in Eq. :eq:`3` yields

.. math:: \Delta \mathbf P_{\textrm{el}} = \frac{e}{2\pi A} \mathrm{Im}\int_{-\pi/c}^{\pi/c} d k_z \sum_n^{\text{occ}} \frac{\partial \theta_{n\mathbf k}}{\partial k_z}.

Because of Eq. :eq:`periodic` we must have
:math:`e^{i\theta_{n\mathbf k}}=e^{i\theta_{n,\mathbf k+\mathbf G}}`
meaning that

.. math:: \theta_{n\mathbf k} = \beta_{n\mathbf k}^{\mathrm{per}}+\mathbf k\cdot \mathbf R_n

where :math:`\beta` is BZ-periodic (and differentiable) in
:math:`\mathbf k`. We thus conclude that for
:math:`H(\lambda=0)=H(\lambda=1)` we have

.. math:: \Delta \mathbf P_{\textrm{el}} = \frac{e}{V_{\textrm{cell}}} \sum_n^{\text{occ}} \mathbf R_n

where :math:`V_\mathrm{cell} = Ac`. This shows that the polarisation
change in direction :math:`\alpha` is only determined up to the
polarisation quantum :math:`(e/V_{\textrm{cell}})L_{\alpha}`.

Eqs. (:eq:`1` - :eq:`2`) invites the interpretation in terms of an absolute
polarisation. However, as previously discussed such a concept is not
well defined. Thus :math:`\mathbf P(\lambda)` only makes sense as a
device to compute the change in polarisation (which when evaluated in
terms of the Berry phase is defined only modulus the polarisation
quantum).

Practical calculations of the polarization
------------------------------------------

We now describe how the Berry phase theory can be used to calculate real
world quantities in practice. Eq. :eq:`2` is slightly rewritten to make
apparent its use of a trace

.. math::
   :label: ndotP

   \mathbf{n}\cdot\mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3} \mathrm{Im}\int_{\mathrm{BZ}}d\mathbf k \, \mathrm{Tr}_\mathrm{occ} \left( \langle u_{nk}^\lambda |\mathbf{n} \cdot \nabla_{\mathbf k}|u_{mk}^\lambda \rangle\right),

where it is understood that the inside of the trace is a matrix in
band-indices :math:`n,m` and that trace is taken over the occupied
manifold of bands. :math:`\mathbf{n}` is a unit-vector along the
direction the polarization is calculated. The derivative of the
Bloch-functions is expanded to first order in :math:`\mathbf{k}`

.. math:: \nabla_{\mathbf k}|u_{m\mathbf{k}}^\lambda \rangle \approx \frac{ |u_{m\mathbf{k}+ \delta \mathbf{k}}^\lambda \rangle-|u_{m\mathbf{k}}^\lambda \rangle}{\delta \mathbf{k}}

leading to the approximate expression for the polarization

.. math:: \mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3} \mathrm{Im}\int_{\mathrm{BZ}_\perp}d\mathbf k_\perp \sum_{\mathbf k_\parallel}\, \mathrm{Tr}_\mathrm{occ} \left( \langle u_{n\mathbf{k}}^\lambda |u_{m\mathbf{k}+\delta\mathbf{k}}^\lambda\rangle - 1 \right).

(Here we have removed :math:`\mathbf{n}\cdot` as it should be clear
that the polarisation along a specific direction is obtained by dotting
with :math:`\mathbf{n}`). In principle, this expression can be
straightforwardly implemented numerically. However, it appears that the
result depend on the (arbitrary) phases of the Bloch states. Eq.
:eq:`ndotP` requires that the :math:`u_{n\mathbf k}` are differentiable
with respect to :math:`\mathbf k`. But the wave functions obtained from
practical DFT codes come with arbitrary phases. To show that the result
is in fact independent of the phases, we use that the logarithm of a
matrix, :math:`S`, which is close to the unit matrix, to first order is

.. math:: \mathrm{ln}(S) \approx (S - I)

which allows us to write

.. math:: \mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3} \mathrm{Im}\int_{\mathrm{BZ}_\perp}d\mathbf k_\perp \sum_{\mathbf k_\parallel}\, \, \mathrm{Tr}_\mathrm{occ} \, \mathrm{ln} \left[\langle u_{n\mathbf{k}}^\lambda |u_{m\mathbf{k}+\delta\mathbf{k}}^\lambda\rangle\right].

Now we can use the fact that the trace of a logarithm of a matrix is
equal to the logarithm of the determinant

.. math:: \mathrm{Tr} \, \mathrm{ln} \, S = \mathrm{ln} \det S

(which can be confirmed by inserting the eigen-representation of
:math:`S`) yielding

.. math:: \mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3} \mathrm{Im}\int_{\mathrm{BZ}_\perp}d\mathbf k_\perp  \, \sum_{\mathbf k_\parallel}\, \mathrm{ln} \, \det_\mathrm{occ} \, \left[\langle u_{n\mathbf{k}}^\lambda |u_{m\mathbf{k}+\delta\mathbf{k}}^\lambda\rangle\right].

Finally we can pull the sum into the logarithm by converting it to a
product

.. math::
   :label: dP_practical

   \mathbf P_{\mathrm{el}}(\lambda) = \frac{e}{8\pi^3} \mathrm{Im}\int_{\mathrm{BZ}_\perp}d\mathbf k_\perp  \, \mathrm{ln} \, \prod_{\mathbf k_\parallel}\, \det_\mathrm{occ} \, \left[\langle u_{n\mathbf{k}}^\lambda |u_{m\mathbf{k}+\delta\mathbf{k}}^\lambda\rangle\right]

This expression shows that the polarization is in fact independent of
the arbitrary phases of the wave functions. It is implemented in the
GPAW code, and one example of its use will be illustrated in the next
section.

The Born charge
---------------

Now we consider the induced polarization when displacing an atom in a
crystal from its equilibrium position. If the atom is ionized and thus
have donated or accepted a finite number of electrons (like in the NaCl
crystal), the induced polarisation can be simply given by the charge of
the ion multiplied by the displacement :math:`\delta \mathbf{R}`

.. math::

   \delta \mathbf{P} = \frac{e Z_\mathrm{ion}}{V_\mathrm{cell}} \delta \mathbf{R}.

Here :math:`Z_\mathrm{ion}` is a number describing the net-charge
associated with the ion. If the electrons are strongly bound to the ion
they will follow the displacement of the ion and :math:`Z` will be
expected to be an integer, however, in the general case where electrons
do not strictly follow the displacement of the ion, :math:`Z` will be a
fractional number known as the Born charge. The Born charge of a given
atom, :math:`a`, in a crystal is a tensor defined as

.. math::
   :label: born

   Z^*_{a,ij} = \frac{V_\mathrm{cell}}{e} \frac{\partial P_{j}}{\partial R_{a,i}}

where :math:`i,j=x,y,x` denote the direction. In this equation it is
understood that atom :math:`a` in all unit cells are displaced such that
the assumption of a periodic perturbation behind Eq.
:eq:`dP\_practical` is satisfied. At this point it is instructive to
recall the definition of the electronic dielectric tensor and
susceptibilites that we have studied so far in the course:

.. math:: \chi^{el}_{ij} = \frac{\partial P^{el}_{j}}{\partial E_i}

and

.. math:: \mathbf \epsilon^{el} = \epsilon_0(1+\mathbf \chi^{el}).

In writing these relations we have suppressed the :math:`q`- and
:math:`\omega`-dependence of the response functions. The important point
is the “el” superscript, which indicates that the induced polarization
is created by the electrons moving in the frozen crystal, i.e without
allowing the atoms to move. To obtain total dielectric tensor and
susceptibilities we must add the ionic part describing the additional
polarization due to the vibrating lattice. The calculation of the ionic
contribution to the dielectric function requires the vibrational
frequencies of the lattice, i.e. the phonons, and the Born charges, as
input. If you would like to see how this goes, consult page 417-419 and
423-424 in GP.

In practice, formula :eq:`born` is evaluated as a finite difference

.. math::

   \frac{\partial \mathbf{P}(\mathbf{R})}{\partial \mathbf{R}} \approx \frac{\mathbf{P}(\mathbf{R} + \delta \mathbf{R}) - \mathbf{P}(\mathbf{R} - \delta \mathbf{R})}{2 \delta \mathbf{R}}.

Finally, we need to use Eq. :eq:`dP_practical` to calculate the
polarisation at a finite displacement of the atoms. However, it is
important to remember that the complex logarithm has a branch cut
which typically lies from :math:`[-\infty, 0]`, which can lead to
discontinuous jumps of the integrand in Eq. :eq:`dP_practical`
yielding unphysical results (the integrand should be continuous). An
example is shown in Fig. :numref:`berry_phase` for the two-dimensional
material MoS\ :math:`_2` where the integral (over
:math:`\mathbf{k}_\perp`) is one-dimensional and therefore can be
easily plotted. Here it is clear that the branch cut of the logarithm
is being crossed leading to discontinuous jumps in the integrand (blue
line).  This can be fixed by comparing neighbouring k-points in the
integrand and adding or subtracting a multiple of :math:`2\pi` to
ensure that the Berry phases change slowly as a function of
:math:`\mathrm{k}_\perp` (orange lines). Using this scheme we find
that two-dimensional MoS\ :math:`_2` in the H-phase has the following
Born charges: :math:`Z^\mathrm{Mo}_{[xx, yy, zz]} = [-1.07, -1.07,
-0.13]` and :math:`Z^\mathrm{S}_{[xx, yy, zz]} = [0.53, 0.53, 0.07]`
(all off-diagonal elements are zero). Now it can be seen that
:math:`Z^\mathrm{S} \approx -Z^\mathrm{Mo} / 2` which is actually a
variant of a deeper principle known as the acoustic sum rule which
says that :math:`\sum^A Z^A_{ij} = 0` (when the net-charge of the
total cell is zero), where :math:`A` runs over all atoms in the unit
cell. It is interesting to note that the Born charges of S are
positive and those of Mo are negative while the opposite is found for
the net charge of the atoms in the equilibrium structure (S takes
electron density from Mo).  This shows that the concept of Born
charges on covalently bonded structures like MoS\ :math:`_2` is highly
non-trivial.

.. figure:: berry_phases.png
   :name: berry_phase

   Calculated Berry phase for MoS\ :math:`_2` showing a discontinuous
   jump as the phase crosses the branch cut of the complex logarithm.
