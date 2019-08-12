from asr.utils import command, option


@command('asr.browser')
@option('--database')
@option('--custom')
@option('--only-figures', is_flag=True,
        help='Dont show browser, just save figures')
def main(database='database.db', custom='asr.utils.custom',
         only_figures=True):
    """Open results in web browser"""
    import subprocess
    from pathlib import Path

    if custom == 'asr.utils.custom':
        custom = Path(__file__).parent / 'utils' / 'custom.py'

    cmd = f'python3 -m ase db {database} -w -M {custom}'
    if only_figures:
        cmd += ' -l'
    print(cmd)
    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit(1)


if __name__ == '__main__':
    main.cli()
