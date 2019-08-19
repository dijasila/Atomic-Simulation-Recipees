from asr.utils import command, option, argument


@command('asr.database.clonetree')
@argument('patterns', nargs=-1, required=False, metavar='PATTERN')
@argument('destination', metavar='DESTDIR')
@argument('source', metavar='SRCDIR')
@option('--run', is_flag=True, help='Actually do something.')
@option('--verbose', '-v', help='Print what\'s being done.')
@option('--copy/--symlink', is_flag=True)
@option('--overwrite-existing', is_flag=True)
@option('--run-only-dirs', is_flag=True)
def main(source, destination, patterns, run=False, verbose=False,
         copy=False, overwrite_existing=False, run_only_dirs=False):
    """Tool for copying or symlinking a tree of files."""
    import fnmatch
    from pathlib import Path
    from typing import List, Tuple
    from click import progressbar
    
    if run_only_dirs:
        assert not run, '--run and --run-only-dirs are mutually exclusive.'

    print(f'Clone {source} to {destination}')
    if patterns:
        string = ', '.join(patterns)
        print(f'Patterns: {string}')
    
    source = Path(source)
    destination = Path(destination)

    if not patterns:
        patterns = ['*']
    
    log: List[Tuple[Path, Path]] = []
    mkdir: List[Path] = []
    errors = []

    def item_show_func(item):
        return str(item)

    with progressbar(source.glob('**/'),
                     label='Searching for files and folders',
                     item_show_func=item_show_func) as bar:
        for srcdir in bar:
            destdir = destination / srcdir.relative_to(source)
            if not destdir.is_dir():
                mkdir.append(destdir)

            for srcfile in srcdir.glob('*'):
                if srcfile.is_file():
                    destfile = destdir / srcfile.name
                    # If file matches any pattern then log it
                    if any([fnmatch.fnmatch(srcfile.name, pattern)
                            for pattern in patterns]):
                        if destfile.is_file():
                            if not overwrite_existing:
                                errors.append(f'{destfile} already exists')
                            continue
                        log.append((srcfile, destfile))

    if len(errors) > 0:
        for error in errors:
            print(error)
        raise AssertionError

    if run or run_only_dirs:
        with progressbar(mkdir,
                         label=f'Creating {len(mkdir)} folders') as bar:
            for destdir in bar:
                destdir.mkdir()
    else:
        if verbose:
            for destdir in mkdir:
                print(f'New folder: {destdir}')
        print(f'Would create {len(mkdir)} folders')

    if copy:
        if run:
            print(f'Copying {len(log)} files')
            with progressbar(log) as bar:
                for srcfile, destfile in bar:
                    destfile.write_bytes(srcfile.read_bytes())
        else:
            if verbose:
                for srcfile, destfile in log:
                    print(f'Copy {srcfile} to {destfile}')
            print(f'Would copy {len(copy)} files')
    else:
        if run:
            with progressbar(log, label=f'symlinking {len(log)} files') as bar:
                for srcfile, destfile in bar:
                    destfile.symlink_to(srcfile.resolve())
        else:
            if verbose:
                for srcfile, destfile in log:
                    print(f'Symlink: {destfile} -> {srcfile}')
            print(f'Would create {len(mkdir)} folders and '
                  f'symlink {len(log)} files')


if __name__ == '__main__':
    main.cli()
