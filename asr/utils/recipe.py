import importlib
from pathlib import Path


class Recipe:

    all_recipes = []

    def __init__(self,
                 name,
                 main=None,
                 postprocessing=None,
                 collect=None,
                 webpanel=None,
                 group=None,
                 creates=None,
                 dependencies=None,
                 resources=None,
                 diskspace=None,
                 restart=None):
        self._name = name
        self._main = main
        self._postprocessing = postprocessing
        self._collect = collect
        self._webpanel = webpanel
        self._group = group
        self._creates = creates
        self._dependencies = dependencies
        self._resources = resources
        self._diskspace = diskspace
        self._restart = restart

    @classmethod
    def frommodule(cls, name, reload=True):
        """Use like: Recipe.frompath('asr.relax')"""
        module = importlib.import_module(f'{name}')

        if reload:
            module = importlib.reload(module)

        kwargs = {}
        if hasattr(module, 'main'):
            kwargs['main'] = module.main
        if hasattr(module, 'postprocessing'):
            kwargs['postprocessing'] = module.postprocessing
        if hasattr(module, 'collect'):
            kwargs['collect'] = module.collect
        if hasattr(module, 'webpanel'):
            kwargs['webpanel'] = module.webpanel
        if hasattr(module, 'group'):
            kwargs['group'] = module.group
        if hasattr(module, 'creates'):
            kwargs['creates'] = module.creates
        if hasattr(module, 'dependencies'):
            kwargs['dependencies'] = module.dependencies
        if hasattr(module, 'resources'):
            kwargs['resources'] = module.resources
        if hasattr(module, 'diskspace'):
            kwargs['diskspace'] = module.diskspace
        if hasattr(module, 'restart'):
            kwargs['restart'] = module.restart

        return cls(**kwargs)

    def main(self, *args, **kwargs):
        if self._main:
            results = self._main(*args, **kwargs)

        if self._postprocessing:
            results.update(self.postprocessing)

        results.update(get_excecution_info(ctx.params))
            
        return results

    def postprocessing(self, *args, **kwargs):
        if hasattr(self.module, 'postprocessing'):
            return self.module.main(*args, **kwargs)
        return NotImplemented

    def collect_data(self, *args, **kwargs):
        if hasattr(self.module, 'collect_data'):
            return self.module.collect_data(*args, **kwargs)
        return NotImplemented

    def webpanel(self, *args, **kwargs):
        if hasattr(self.module, 'webpanel'):
            return self.module.main(*args, **kwargs)
        return NotImplemented

    def creates(self, *args, **kwargs):
        if hasattr(self.module, 'creates'):
            if callable(self.module.creates):
                return self.module.creates(*args, **kwargs)
            return self.module.creates
        return NotImplemented

    def resources(self, *args, **kwargs):
        if hasattr(self.module, 'resources'):
            if callable(self.module.resources):
                return self.module.resources(*args, **kwargs)
            return self.module.resources
        return NotImplemented

    def restart(self, *args, **kwargs):
        if hasattr(self.module, 'restart'):
            if callable(self.module.restart):
                return self.module.restart(*args, **kwargs)
            return self.module.restart
        return NotImplemented

    def diskspace(self, *args, **kwargs):
        if hasattr(self.module, 'diskspace'):
            if callable(self.module.diskspace):
                return self.module.diskspace(*args, **kwargs)
            return self.module.diskspace
        return NotImplemented

    def done(self):
        name = self.name[4:]
        creates = [f'results_{name}.json']
        if self.creates:
            creates += self.creates

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
