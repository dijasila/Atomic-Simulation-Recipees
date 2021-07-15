from pathlib import Path


def make_property(name):
    def getter(self):
        return self.data[name]

    def setter(self, value):
        self.data[name] = value

    return property(getter, setter)


class WebPanel:

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

    columns = make_property('columns')
    title = make_property('title')
    plot_descriptions = make_property('plot_descriptions')
    sort = make_property('sort')
    id = make_property('id')

    def __getitem__(self, item):  # noqa
        return self.data[item]

    def get(self, item, default):
        return self.data.get(item, default)

    def update(self, dct):
        self.data.update(dct)

    def items(self):
        return self.data.items

    def __contains__(self, key):  # noqa
        return key in self.data

    def __repr__(self):  # noqa
        return (f'WebPanel(title="{self.title}",'
                f'columns={self.columns},sort={self.sort},...)')
