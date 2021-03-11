from asr.core import command, option


@command(module='asr.test.test_options',
         creates=['pippo.txt'])
@option('--param', type=int)
def pippo(param: int=0):
    with open('pippo.txt', 'w') as pip:
        print(param, file=pip)


@command(module='asr.test.test_options',
         creates=['baudo.txt'])
@option('--baudo', type=int)
def baudo(baudo: int=0):
    with open('baudo.txt', 'w') as bau:
        print(baudo, file=bau)
'''
@command(module='asr.test.test_options',
         creates=['pippo.txt'])
@option('--roba', type=int)
def main(roba: int=100):
    with open('main.txt') as mn:
        print(roba, file=mn)
'''


if __name__ == '__main__':
    main.cli()
