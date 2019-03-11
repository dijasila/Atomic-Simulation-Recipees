from pathlib import Path


class Recipe:

    def __init__(self, module):

        # Known attributes
        attributes = ['group', 'short_description', 'description', 'parser',
                      'dependencies', 'creates', 'resources',
                      'diskspace', 'restart', 'main']

        for attr in attributes:
            tmp = getattr(module, attr, None)
            setattr(self, attr, tmp)

        # Set default values
        if not self.group:
            self.group = ''
        if not self.short_description:
            self.short_description = ''

        self.name = Path(module.__file__).with_suffix('').name
