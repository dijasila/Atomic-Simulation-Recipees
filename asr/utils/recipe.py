import importlib


class Recipe:

    all_recipes = []
    
    known_attributes = ['main', 'collect_data',
                        'webpanel', 'group', 'creates',
                        'dependencies', 'resources', 'diskspace',
                        'restart']

    def __init__(self, module):
        self.__name__ = module.__name__
        self.implemented_attributes = []
        for attr in Recipe.known_attributes:
            if hasattr(module, attr):
                self.implemented_attributes.append(attr)
                setattr(self, attr, getattr(module, attr))

    # Alternative contructors
    @classmethod
    def frompath(cls, name):
        module = importlib.import_module(f'{name}')
        return cls(module)

    def __repr__(self):
        from asr.utils.cli import format
        from inspect import signature
        msg = [['Detailed recipe info'],
               ['--------------------']]
        msg += [['Name:', f'{self.__name__}']]
        exclude = ['main']

        if self.main:
            msg += ['MAIN']
            msg += ['=' * 4]
            ctx = self.main.make_context('', [])
            
            msg += [self.main.get_help(ctx) + '\n' * 2]
        else:
            msg += ['ERROR: main function not implemented']

        if self.collect_data:
            msg += ['COLLECT_DATA', '=' * 12]
            sig = str(signature(self.collect_data))
            sig2 = str(signature(template.collect_data))
            if sig == sig2:
                msg += [f'Signature collect_data{sig} is correct' + '\n' * 2]
            else:
                msg += [f'ERROR: wrong signature collect_data{sig}' + '\n' * 2]
        else:
            msg += ['ERROR: collect_data function not implemented']
            
        if self.webpanel:
            msg += ['WEBPANEL', '=' * 8]
            sig = str(signature(self.webpanel))
            sig2 = str(signature(template.webpanel))
            if sig == sig2:
                msg += [f'Signature collect_data{sig} is correct' + '\n' * 2]
            else:
                msg += [f'ERROR: wrong signature collect_data{sig}' + '\n' * 2]
        else:
            msg += ['ERROR: collect_data function not implemented']
            
        check_signatures = ['collect_data', 'webpanel']
        
        notimp = [[''], ['Issues'], ['------']]
        for attr in self.known_attributes:
            if attr in exclude:
                continue
            
            var = getattr(self, attr)
            if var:
                if attr in check_signatures:
                    sig = signature(var)
                    if callable(var):
                        msg += [[f'{attr}:', f'function{str(sig)}']]
                else:
                    msg += [[f'{attr}:', f'{var}']]

            else:
                notimp += [[f'{attr}', f'NotImplemented']]
                
        return format(msg + notimp)
        
    
for attr in Recipe.known_attributes:
    setattr(Recipe, attr, None)

template = Recipe.frompath('asr.utils.something')

    
if __name__ == '__main__':
    r = Recipe.frompath('asr.bandstructure')
