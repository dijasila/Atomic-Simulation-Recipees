from pathlib import Path


class WebPanel:

    def __init__(self, title, columns=None, plot_descriptions=None,
                 subpanel=False, sort=99, id=None):
        self.title = title
        self.columns = columns if columns is not None else [[], []]
        self.plot_descriptions = plot_descriptions if plot_descriptions is not None else []
        self.sort = sort
        self.id = id if id is not None else title

        # check if the panel title belongs to the sub-panel group
        if subpanel:
            self.subpanel = subpanel

        self.data = dict(
            columns=columns,
            title=title,
            plot_descriptions=plot_descriptions,
            sort=sort,
            id=id,
        )

    def __getitem__(self, item):  # noqa
        return self.__dict__[item]

    def get(self, item, default):
        return self.__dict__.get(item, default)

    def update(self, dct):
        self.__dict__.update(dct)

    def items(self):
        return self.__dict__.items()

    def __contains__(self, key):  # noqa
        return key in self.__dict__

    def __str__(self):  # noqa
        return (f'WebPanel(title="{self.title}",'
                f'columns={self.columns},sort={self.sort},...)')

    def __repr__(self):  # noqa
        return str(self)

    def render(self) -> str:
        from jinja2 import Template
        path = Path(__file__).parent / 'templates/webpanel.html'
        return Template(path.read_text()).render(webpanel=self)


SummaryLayout = [
    WebPanel(title='Summary'),
    WebPanel(title='Thermodynamic stability'),
    WebPanel(title='Stiffness tensor'),
    WebPanel(title='Phonons'),
]
