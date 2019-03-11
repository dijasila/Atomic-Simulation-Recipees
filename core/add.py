from pathlib import Path
from myrecipes.utilities import get_home_folders, write_home_folders


def add(folder):
    folder = Path(folder).absolute()
    folders = get_home_folders()
    if folder in folders:
        print('Collection already present!')
        return
    folders = folders + [folder]
    write_home_folders(folders)
    

def get_parser():
    import argparse
    desc = 'Add recipes to cookbook'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('folder')
    return parser


short_description = 'Add recipes to collection'
parser = get_parser()


def main(args=None):
    add(args['folder'])


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())
    main()
