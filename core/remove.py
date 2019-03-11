from pathlib import Path
from myrecipes.utilities import get_home_folders, write_home_folders


def remove(folder):
    folder = Path(folder).absolute()
    folders = get_home_folders()
    newfolders = []
    for folder1 in folders:
        if folder1 == folder:
            continue
        newfolders.append(folder1)

    if len(newfolders) == len(folders):
        print('Did not find collection!')
        return
        
    write_home_folders(newfolders)


def get_parser():
    import argparse
    desc = 'Remove recipies from cookbook'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('folder', type=str)
    return parser


parser = get_parser()
short_description = 'Remove recipes from collection'


def main(args=None):
    if args is None:
        parser = get_parser()
        args = vars(parser.parse_args())
    remove(**args)


if __name__ == '__main__':
    main()
