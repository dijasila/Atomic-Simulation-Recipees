Installation
============
To install ASR first clone the code and pip-install the code::

  $ cd ~ && git clone https://gitlab.com/mortengjerding/asr.git
  $ python3 -m pip install -e ~/asr


ASE has to be installed manually since we need the newest version::

  $ python3 -m pip install git+https://gitlab.com/ase/ase.git

Also if you don't already have GPAW installed you can get it with::

  $ python3 -m pip install git+https://gitlab.com/gpaw/gpaw.git

Install a database of reference energies to calculate HOF and convex hull. Here 
we use a database of one- and component-structures from OQMD::

  $ cd ~ && wget https://cmr.fysik.dtu.dk/_downloads/oqmd12.db
  $ echo 'export ASR_REFERENCES=~/oqmd12.db' >> ~/.bashrc

We do relaxations with the D3 van-der-Waals contribution. To install the van 
der Waals functional DFTD3 do::

  $ mkdir ~/DFTD3 && cd ~/DFTD3
  $ wget http://chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz
  $ tar -zxf dftd3.tgz
  $ make
  $ echo 'export ASE_DFTD3_COMMAND=$HOME/DFTD3/dftd3' >> ~/.bashrc
  $ source ~/.bashrc

To make Bader analysis we use another program. Download the executable for Bader 
analysis and put in path (this is for Linux, find the appropriate executable)::

  $ cd ~ && mkdir baderext && cd baderext
  $ wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz
  $ tar -zxf bader_lnx_64.tar.gz
  $ echo  'export PATH=~/baderext:$PATH' >> ~/.bashrc

Additionally, you might also want
* myqueue

if you want to run jobs on a computer-cluster.

Finally, test the package with::
  $ asr test
