"""Run workflow module.."""
from typing import Union
from asr.core import command, option, ASRResult


@command('asr.workflow',
         save_results_file=False)
@option("--include", help="Comma separated list of includes.", type=str)
@option("--exclude", help="Comma separated list of exludes.", type=str)
def main(include: Union[str, None] = None,
         exclude: Union[str, None] = None) -> ASRResult:
    """Run a full material property workflow."""
    import urllib.request
    from asr.core import get_recipes

    if include is not None:
        include = include.split(",")

    if exclude is not None:
        exclude = exclude.split(",")

    recipes = get_recipes()
    order = [
        "asr.relax",
        "asr.gs",
        "asr.convex_hull",
        "asr.phonons",
        "asr.setup_strains",
        "asr.magnetic_anisotropy",
        "asr.stiffness",
        "asr.emasses",
        "asr.pdos",
        "asr.bandstructure",
        "asr.projected_bandstructure",
        "asr.polarizability",
        "asr.plasmafrequency",
        "asr.fermisurface",
        "asr.borncharges",
        "asr.piezoelectrictensor"
        "asr.infrared_polarizability",
        "asr.push",
        "asr.raman",
        "asr.bse"
    ]
    url = ("https://cmr.fysik.dtu.dk/_downloads/"
           "ebe5e92dd4d83fd16999ce911c8527ab/oqmd12.db")
    urllib.request.urlretrieve(url, "oqmd12.db")

    extra_args = {"asr.convex_hull": {"databases": ["oqmd12.db"]}}

    def filterfunc(x):
        if exclude and x.name in exclude:
            return False
        if include and x.name not in include:
            return False
        if x.name not in order:
            return False
        return True

    recipes = filter(filterfunc, recipes)
    recipes = sorted(recipes, key=order.index)
    for recipe in recipes:
        kwargs = extra_args.get(recipe.name, {})
        recipe(**kwargs)


if __name__ == '__main__':
    main.cli()
