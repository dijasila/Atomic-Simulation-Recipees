from asr.utils import get_recipes

# Can we get all recipes?
recipes = get_recipes()

for recipe in recipes:
    """Call all main functions with --help"""

    if not recipe.main:
        continue

    try:
        func = recipe.main
        func(args=['--help'])
    except Exception:
        print(f'Problem in function {recipe.__name__}.main '
              'when called with --help')
        raise


for recipe in recipes:
    """Make sure that the group property is implemented"""
    if not recipe.group:
        continue

    assert recipe.group in ['structure', 'property', 'postprocessing',
                            'setup'], \
        (f'Group {recipe.__name__} not known!')


for recipe in recipes:
    """Make sure that the correct _asr_command is being used"""
    if not recipe.main:
        continue
    
    try:
        assert hasattr(recipe.main, '_asr_command')
    except AssertionError:
        msg = ('Dont use @click.command! Please use '
               'the "from asr.utils import command" '
               'in stead')
        raise AssertionError(msg)
