import datetime
import typing
import dataclasses
import pathlib

from .config import find_root


def construct_metadata(
        created: typing.Optional[datetime.datetime] = None,
        modified: typing.Optional[datetime.datetime] = None,
        directory: typing.Optional[str] = None,
) -> 'Metadata':

    if created is None:
        created = datetime.datetime.now()

    if modified is None:
        modified = created

    if directory is None:
        directory = str(pathlib.Path('.').absolute().relative_to(find_root()))

    return Metadata(
        created=created,
        modified=modified,
        directory=directory,
    )


@dataclasses.dataclass
class Metadata:
    """Class representing record metadata.

    Attributes
    ----------
    created : Record creation date.
    modified : Record modification date.
    directory : Record directory.
    """

    created: typing.Optional[datetime.datetime] = None
    modified: typing.Optional[datetime.datetime] = None
    directory: typing.Optional[str] = None

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            lines.append(f'{key}={value}')
        return '\n'.join(lines)


def register_metadata():

    def wrap(func):

        def wrapped(*args, **kwargs):
            record = func(*args, **kwargs)
            record.metadata = construct_metadata()
            return record

        return wrapped

    return wrap
