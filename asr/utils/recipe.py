import importlib


class Recipe:

    all_recipes = []
    
    known_attributes = ['main', 'collect_data',
                        'webpanel', 'group', 'creates',
                        'dependencies', 'resources', 'diskspace',
                        'restart']

    def __init__(self, module):
        self.__name__ = module.__name__
        print(f'Initializing recipe {self.__name__}')
        self.implemented_attributes = []
        for attr in Recipe.known_attributes:
            if hasattr(module, attr):
                self.implemented_attributes.append(attr)
                setattr(self, attr, getattr(module, attr))

    # Alternative contructors
    @classmethod
    def frompath(cls, name, reload=False):
        module = importlib.import_module(f'{name}')
        if reload:
            module = importlib.reload(module)
        return cls(module)

    def run(self):
        return self.main(args=[])


for attr in Recipe.known_attributes:

    def func():
        raise NotImplementedError
    
    setattr(Recipe, attr, func)
