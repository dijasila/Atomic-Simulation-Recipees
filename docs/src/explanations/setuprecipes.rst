The setup recipes
-----------------

ASR also includes some special `setup` recipes. These recipes are
meant to give the user some easy tools to setup atomic
structures. Here we provide some explanations of their usage.

* The `setup.magnetize` recipe is useful if you don't know the
  magnetic configuration of the material you are currently
  investigation. It sets up non-magnetic (nm), magnetic (fm) and
  anti-ferro magnetic (afm, only for exactly two magnetic atoms in the
  unit cell) configurations of the inital magnetic moments of the
  structure in new subfolders `nm/` `fm/` and `afm`, respectively. For
  another example of using the magnetize recipe see the "Advanced
  Example: Make a screening study" section. For more information see
  `asr help setup.magnetize`
* The `setup.decorate` recipe is useful if you want to create new
  atomic that are similar to an existing atomic structure. The
  decorate recipe contains a table describing the likelyhood of two
  atoms to be substituted. By default the decorate recipe creates a
  new ASE database with the decorated atomic structure (including
  itself). For more information see `asr help setup.decorate`.
* The `setup.unpackdatabase` recipe is useful if you have a database
  of materials that you wish to conduct some calculations on. By
  default, running `asr run setup.unpackdatabase` creates a new folder
  `tree/` in the current directory with all mateirals distributed
  according to the following folder structure
  `tree/{stoi}/{spg}/{formula:metal}-{stoi}-{spg}-{wyck}-{uid}` where
  `stoi` is the stoichiometry, `spg` is the space group number, `wyck`
  are the alphabetically sorted unique Wyckoff positions of the
  materials, `formula:metal` is the chemical formula sorted after
  metal atoms first and `uid` is a unique identifier to avoid
  collisions between materials that would otherwise end up in the same
  folder. For another example of using the unpackdatabase recipe see
  the "Advanced Example: Make a screening study" section. For more
  information see `asr run "setup.unpackdatabase -h"`.
* The `setup.params` recipe is useful as it makes a `params.json` file
  containing the default parameters of all recipes. This makes it
  possible to modify the input parameters used by each recipe. See the
  "Change default settings in scripts" section for more information on
  how this works.
* The `setup.scanparams` recipe is useful if you want to conduct a
  convergence study of a given recipe. As argument it takes a number
  of different values for the input arguments to a recipe and
  generates a series of folders that contain a `params.json` file with
  a specific combination of those parameters. When you are done with
  you calculations you can collect the data in the folders and plot
  them in the browser.
