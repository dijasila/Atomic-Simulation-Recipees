import importlib
from pathlib import Path


class Recipe:

    all_recipes = []

    known_attributes = ['main', 'postprocessing',
                        'collect_data', 'webpanel', 'group', 'creates',
                        'dependencies', 'resources', 'diskspace',
                        'restart']

    def __init__(self, module):
        self.name = module.__name__
        self.module = module

        for attr in Recipe.known_attributes:
            if hasattr(module, attr):
                setattr(self, attr, getattr(module, attr))

    def __contains__(self, item):
        if hasattr(self.module, item):
            return True
        return False

    @classmethod
    def frompath(cls, name, reload=True):
        """Use like: Recipe.frompath('asr.relax')"""
        module = importlib.import_module(f'{name}')
        if reload:
            module = importlib.reload(module)
        return cls(module)

    # def main(self, *args, **kwargs):
    #     if hasattr(self.module, 'main'):
    #         print(args, kwargs, self.name, self.module,
    #               self.module.main)
    #         print(dir(self.module.main))
    #         return self.module.main(*args, **kwargs)
    #     return NotImplemented

    # def postprocessing(self, *args, **kwargs):
    #     if hasattr(self.module, 'postprocessing'):
    #         return self.module.main(*args, **kwargs)
    #     return NotImplemented

    # def collect_data(self, *args, **kwargs):
    #     if hasattr(self.module, 'collect_data'):
    #         return self.module.collect_data(*args, **kwargs)
    #     return NotImplemented
        
    # def webpanel(self, *args, **kwargs):
    #     if hasattr(self.module, 'webpanel'):
    #         return self.module.main(*args, **kwargs)
    #     return NotImplemented

    # def creates(self, *args, **kwargs):
    #     if hasattr(self.module, 'creates'):
    #         if callable(self.module.creates):
    #             return self.module.creates(*args, **kwargs)
    #         return self.module.creates
    #     return NotImplemented

    # def resources(self, *args, **kwargs):
    #     if hasattr(self.module, 'resources'):
    #         if callable(self.module.resources):
    #             return self.module.resources(*args, **kwargs)
    #         return self.module.resources
    #     return NotImplemented

    # def restart(self, *args, **kwargs):
    #     if hasattr(self.module, 'restart'):
    #         if callable(self.module.restart):
    #             return self.module.restart(*args, **kwargs)
    #         return self.module.restart
    #     return NotImplemented

    # def diskspace(self, *args, **kwargs):
    #     if hasattr(self.module, 'diskspace'):
    #         if callable(self.module.diskspace):
    #             return self.module.diskspace(*args, **kwargs)
    #         return self.module.diskspace
    #     return NotImplemented

    def done(self):
        name = self.name[4:]
        creates = [f'results_{name}.json']
        if self.creates:
            creates += self.creates
        # modulecreates = self.creates()
        # if modulecreates is not NotImplemented:
        #     creates += self.creates()

        for file in creates:
            if not Path(file).exists():
                return False
        return True

    def run(self, args=None):
        if args is None:
            args = []
        return self.main(args=args)

    def collect(self, atoms):
        kvp = {}
        key_descriptions = {}
        data = {}
        if self.done():
            if self.collect_data:
                kvp, key_descriptions, data = self.collect_data(atoms)

            name = self.name[4:]
            resultfile = Path(f'results_{name}.json')
            from ase.io import jsonio
            results = jsonio.decode(resultfile.read_text())
            key = f'results_{name}'
            msg = f'{self.name}: You cannot put a {key} in data'
            assert key not in data, msg
            data[key] = results

        return kvp, key_descriptions, data


for attr in Recipe.known_attributes:
    setattr(Recipe, attr, None)
