Advanced Example: Make a screening study
----------------------------------------
A screening study what we call a simultaneous automatic study of many materials. ASR
has a set of tools to make such studies easy to handle. Suppose we have an ASE
database that contain many atomic structures. In this case we take OQMD12 database
that contain all unary and binary compounds on the convex hull.

The first thing we do is to get the database::

  $ mkdir ~/oqmd12 && cd ~/oqmd12
  $ wget https://cmr.fysik.dtu.dk/_downloads/oqmd12.db

We then use the `unpackdatabase` function of ASR to unpack the database into a
directory tree::

  $ asr run setup.unpackdatabase oqmd12.db -s u=False --run

(we have made the selection `u=False` since we are not interested in the DFT+U values).
This function produces a new folder `~oqmd12/tree/` where you can find the tree. 
To see the contents of the tree it is recommended to use the linux command `tree`::

  $ tree tree/

You will see that the unpacking of the database has produced many `unrelaxed.json`
files that contain the unrelaxed atomic structures. Because we don't know the
magnetic structure of the materials we also want to sample different magnetic structOBures.
This can be done with the `magnetize` function of asr::

$ asr run setup.magnetize in */*/*/*/

We use the `run` function because that gives us the option to deal with many folders
at once. You are now ready to run a
workflow on the entire tree. ASR has a `workflow` function that let's us build
workflows based on the recipes in ASR. This function is meant as an help to
start on new recipes. Familiarize yourself with the function by
running `asr workflow -h`. The help shows that it is possible to create a
workflow by::

  $ asr workflow -t asr.relax,asr.bandstructure,asr.convex_hull --doforstable asr.bandstructure > workflow.py
  $ cat worflow.py
  from myqueue.task import task


  def is_stable():
      # Example of function that looks at the heat of formation
      # and returns True if the material is stable
      from asr.utils import read_json
      from pathlib import Path
      fname = 'results_convex_hull.json'
      if not Path(fname).is_file():
          return False

      data = read_json(fname)
      if data['hform'] < 0.05:
          return True
      return False


  def create_tasks():
      tasks = []
      tasks += [task('asr.relax@8:xeon8:10h')]
      tasks += [task('asr.structureinfo@1:10m')]
      tasks += [task('asr.gs@8:10h', deps='asr.structureinfo')]
      tasks += [task('asr.gaps@1:10m', deps='asr.structureinfo,asr.gs')]
      tasks += [task('asr.convex_hull@1:10m', deps='asr.structureinfo,asr.gs')]
      if is_stable():
          tasks += [task('asr.bandstructure@1:10m', deps='asr.structureinfo,asr.gaps,asr.gs')]

      return tasks

This workflow relaxes the structures and if the `convex_hull` recipe calculates
low heat of formation the workflow will make sure that the bandstructure is
calculated. We now ask `myqueue` what jobs it wants to run.::

  $ mq workflow -z workflow.py tree/*/*/*/*/

To submit the jobs simply remove the `-z`, and run the command again.

For complex workflows, like the one above where we have to check the stability of
materials which has to wait until the `convex_hull` recipe has finished, the 
`mq workflow` function should have to be run periodically to check for new tasks.
In this case it is smart to set up a crontab to do the work for you. 
To do this write::

  $ crontab -e

choose your editor and put the line 
`*/5 * * * * . ~/.bashrc; cd ~/oqmd12; mq kick; mq workflow workflow.py tree/*/*/*/*/`
into the file. This will restart any timeout jobs and run the workflow command 
to see if any new tasks should be spawned with a 5 minute interval. 

Recommended Procedures
=======================
The tools of ASR can be combined to perform complicated tasks with little
effort. Below you will find the recommended procedures to perform common
tasks within the ASR framework.


Make a convergence study
------------------------
When you have created a new recipe it is highly likely that you would have to
make a convergence study of the parameters in your such that you have proof that
your choice of parameters are converged. The tools of ASR makes it easier to
conduct such convergence studies. ASR has a built-in database with materials
that could be relevant to investigate in your convergence tests. These materials
can be retrieved using the `setup.materials` recipe. See
`asr help setup.materials` for more information. For example, to convergence
check the parameters of `asr.relax` you can do the following.::


  $ mkdir convergence-test && cd convergence-test
  $ asr run setup.materials
  $ asr run setup.unpackdatabase materials.json --tree-structure materials/{formula:metal} --run
  $ cd materials/
  $ asr run setup.scanparams asr.relax:ecut 600 700 800 asr.relax:kptdensity 4 5 6 in */
  $ mq submit asr.relax@24:10h */*/


When the calculations are done you can collect all results into a database and
inspect them::

  $ cd convergence-test
  $ asr run collect */*/
  $ asr run browser
