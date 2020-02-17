from asr.core import command


@command('asr.workflow')
def main():
    """Run a full material property workflow."""

    import urllib.request
    from asr.core import get_recipes

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
    recipes = filter(lambda x: x in order, recipes)
    recipes = sorted(recipes, key=order.index)
    for recipe in recipes:
        kwargs = extra_args.get(recipe.name, {})
        recipe(**kwargs)


if __name__ == '__main__':
    main.cli()
