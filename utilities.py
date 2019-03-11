from pathlib import Path
from typing import List


def get_folder_file():
    return Path.home() / '.myrecipes' / 'folders.txt'


def get_home_folders() -> List[Path]:
    path = get_folder_file()
    if path.is_file():
        folders = []
        for f in path.read_text().splitlines():
            folder = Path(f)
            if folder.is_dir():
                folders.append(folder)
        return folders
    else:
        return [Path(__file__).parent / 'core',
                Path(__file__).parent / 'recipes']


def write_home_folders(folders):
    folderfile = get_folder_file()
    folderfile.write_text('\n'.join(str(folder)
                                    for folder in folders) +
                          '\n')

