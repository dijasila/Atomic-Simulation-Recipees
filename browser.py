import click
from functools import partial
option = partial(click.option, show_default=True)


@click.command()
@option('--database', default='database.db')
@option('--custom', default='asr.custom')
def main(database, custom):
    """Open results in web browser"""
    import subprocess
    from pathlib import Path
    
    if custom == 'asr.custom':
        custom = Path(__file__).parent / 'custom.py'

    cmd = f'ase db {database} -w -M {custom}'
    print(cmd)
    subprocess.run(cmd.split())


group = 'Postprocessing'

if __name__ == '__main__':
    main()
