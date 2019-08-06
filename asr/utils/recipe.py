import importlib
import click
from pathlib import Path
from asr.utils import get_execution_info

def recipe(*args, **kwargs):
    pass


def option(f, *args, default=None, help=None):
    pass

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
        self.name = name
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

    def __call__(self, *args, **kwargs):
        # When we call a recipe we will assume that you mean to
        # execute the recipes main function
        return self.main(*args, **kwargs)

    @property
    def creates(self):
        creates = [f'results_{self.name}.json']
        if self._creates:
            creates += self._creates
        return creates

    @property
    def dependencies(self):
        return self._dependencies or []

    @property
    def resources(self):
        resources = '1:10m'
        if self._resources:
            if callable(self._resources):
                resources = self._resources()
            else:
                resources = self._resources
        return resources

    def cli(self, *args, **kwargs):
        if self._cli:
            return self._cli(*args, **kwargs)

    def main(self, *args, **kwargs):
        results = {}
        if self._main:
            import inspect
            sig = inspect.signature(self._main)
            storedargs = {key: value.default for key, value
                          in sig.parameters.items() if value.default
                          is not inspect.Parameter.empty}
            storedargs.update(dict(sig.bind(*args, **kwargs).arguments))
            mainresults = self._main(*args, **kwargs)
            results['__params__'] = storedargs

            if mainresults:
                results.update(mainresults)

        if self._postprocessing:
            postresults = self._postprocessing()
            if postresults:
                results.update(postresults)

        return results

    def collect_data(self, *args, **kwargs):
        if hasattr(self.module, 'collect_data'):
            return self.module.collect_data(*args, **kwargs)
        return NotImplemented

    def webpanel(self, *args, **kwargs):
        if hasattr(self.module, 'webpanel'):
            return self.module.main(*args, **kwargs)
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

    def __str__(self):
        info = []
        info.append(f'Name: {self.name}')
        info.append(f'Creates: {self.creates}')
        info = '\n'.join(info)
        return info
    

if __name__ == '__main__':
    def save_params(func):

        def wrapper(*args, **kwargs):
            from sys import argv
            results = func(*args, **kwargs)
            results['__params__'] = kwargs
            return results

        return wrapper

    # @save_params
    def main(a, b=3):
        addb = a + b
        result = {'addb': addb}
        return result
        
    def postprocessing():
        print('Executed post processing!')
        return {'postprocessing': 'Failed :-(!'}

    def webpanel():
        pass

    def collect():
        pass

    testrecipe = Recipe('testrecipe',
                        main=main,
                        postprocessing=postprocessing)
    relax = Recipe.frommodule('asr.relax')

    print(testrecipe)
    results = testrecipe(6)
    print(results)
