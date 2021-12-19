import pathlib
from typing import Dict, Any


def get_old_defaults() -> Dict[str, Any]:
    """Get old-master recipe defaults, used in migration and conversion tools.

    Returns
    -------
    Dict[str, Any]
        A dict with keys=instruction-name, value=defaults-dict.
    """
    from .utils import (
        read_file,
        add_main_to_name_if_missing,
        fix_recipe_name_if_recipe_has_been_moved,
    )
    from .serialize import JSONSerializer

    PATH = pathlib.Path(__file__).parent / "old_resultfile_defaults.json"
    TMP_DEFAULTS = JSONSerializer().deserialize(read_file(PATH))
    OLD_DEFAULTS: Dict[str, Dict[str, Any]] = {}
    for tmpkey, value in TMP_DEFAULTS.items():
        OLD_DEFAULTS[
            add_main_to_name_if_missing(
                fix_recipe_name_if_recipe_has_been_moved(tmpkey)
            )
        ] = value
    return OLD_DEFAULTS
