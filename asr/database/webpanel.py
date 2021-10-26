from collections.abc import Mapping


class WebPanel(Mapping):
    def __init__(self, title, columns=None, plot_descriptions=None, sort=99, id=None):

        if plot_descriptions is None:
            plot_descriptions = []

        if columns is None:
            columns = [[], []]

        if id is None:
            id = title

        self.data = dict(
            columns=columns,
            title=title,
            plot_descriptions=plot_descriptions,
            sort=sort,
            id=id,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return (
            f'WebPanel(title="{self.title}",'
            f"columns={self.columns},sort={self.sort},...)"
        )
