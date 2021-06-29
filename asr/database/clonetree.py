"""Clone folder tree."""


# @command('asr.database.clonetree')
# @argument('patterns', nargs=-1, required=False, metavar='PATTERN',
#           type=str)
# @argument('destination', metavar='DESTDIR', type=str)
# @argument('source', metavar='SRCDIR', type=str)
# @option('--copy/--symlink', is_flag=True)
# @option('--map-files', type=str)
# @option('--dont-contain', type=str)
# @option('--must-contain', type=str)
# @option('--dry-run', type=bool)
# @option('--glob-pattern', type=str)
def main(source: str, destination: str, patterns: str,
         copy: bool = False, map_files: str = None, dont_contain: str = None,
         must_contain: str = None, dry_run: bool = False,
         glob_pattern: str = '**/'):
    """Tool for copying or symlinking a tree of files."""
    import fnmatch
    from pathlib import Path
    from typing import List, Tuple
    from click import progressbar
    from os import makedirs

    print(f'Clone {source} to {destination}')
    if patterns:
        string = ', '.join(patterns)
        print(f'Patterns: {string}')

    if must_contain:
        must_contain = must_contain.split(',')
    else:
        must_contain = []

    if dont_contain:
        dont_contain = dont_contain.split(',')
    else:
        dont_contain = []

    source = Path(source)
    destination = Path(destination)

    if not patterns:
        patterns = ['*']

    log: List[Tuple[Path, Path]] = []
    mkdir: List[Path] = []
    errors = []

    def item_show_func(item):
        return str(item)

    with progressbar(source.glob(glob_pattern),
                     label='Searching for files and folders',
                     item_show_func=item_show_func) as bar:
        for srcdir in bar:
            destdir = destination / srcdir.relative_to(source)

            dirfiles = list(srcdir.glob('*'))
            dirfilenames = [srcfile.name for srcfile in dirfiles]

            # Directory has to contain all these files
            contains = [True if name in dirfilenames else False
                        for name in must_contain]
            if not all(contains):
                continue

            # Directory must not contain these files
            not_contains = [True if name in dirfilenames else False
                            for name in dont_contain]
            if any(not_contains):
                continue

            for srcfile in dirfiles:
                if srcfile.is_file():
                    destfile = destdir / srcfile.name
                    if any([fnmatch.fnmatch(srcfile.name, pattern)
                            for pattern in patterns]):
                        if destfile.is_file():
                            continue
                        log.append((srcfile, destfile))

            mkdir.append(destdir)

    if len(errors) > 0:
        for error in errors:
            print(error)
        raise AssertionError

    if not dry_run:
        with progressbar(mkdir,
                         label=f'Creating {len(mkdir)} folders') as bar:
            for destdir in bar:
                if destdir.is_dir():
                    continue
                makedirs(str(destdir))
    else:
        print(f'Would create {len(mkdir)} folders')

    if copy:
        if not dry_run:
            print(f'Copying {len(log)} files')
            with progressbar(log) as bar:
                for srcfile, destfile in bar:
                    destfile.write_bytes(srcfile.read_bytes())
        else:
            print(f'Would copy {len(copy)} files')
    else:
        if not dry_run:
            with progressbar(log, label=f'symlinking {len(log)} files') as bar:
                for srcfile, destfile in bar:
                    destfile.symlink_to(srcfile.resolve())
        else:
            print(f'Would create {len(mkdir)} folders and '
                  f'symlink {len(log)} files')

    # Finally we allow for some postprocessing or massaging of the
    # created folders if some files have to be swapped around
    from asr.core import chdir
    if map_files:
        mapping = [tmp.split('->') for tmp in map_files.split(',')]

        with progressbar(mkdir, label='Remapping files') as bar:
            for destdir in bar:
                with chdir(destdir):
                    for orig, replace in mapping:
                        for src in Path('.').glob('*'):
                            if orig not in src.name:
                                continue
                            dest = Path(src.name.replace(orig, replace))
                            if Path(dest).is_file():
                                Path(dest).unlink()
                            dest.symlink_to(src.resolve())


if __name__ == '__main__':
    main.cli()
