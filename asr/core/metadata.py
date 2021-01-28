import datetime
import typing


def construct_metadata(
        created: typing.Optional[datetime.datetime] = None,
        modified: typing.Optional[datetime.datetime] = None,
) -> 'Metadata':

    if created is None:
        created = datetime.datetime.now()

    if modified is None:
        modified = created

    return Metadata(created=created, modified=modified)


class Metadata:

    def __init__(
            self,
            created: typing.Optional[datetime.datetime] = None,
            modified: typing.Optional[datetime.datetime] = None,
    ):

        self._created = created
        self._modified = modified

    @property
    def created(self):
        """Creation date."""
        return self._created

    @property
    def modified(self):
        """Modification date."""
        return self._modified

    def __str__(self):
        return f'Metadata(created={self.created}, modified={self.modified})'
