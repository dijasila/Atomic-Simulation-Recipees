class HTMLStringFormat:
    def __init__(self, html_format):
        """
        html_format: Any valid html format element
        """
        self.html_format = html_format

    def void_tag(self):
        """
        Add an empty element or are so-called void elements i.e. tags
        that stand independently and may not even contain attributes like <br>
        which creates a line break.
        """
        def void(text: str, end: bool = True):
            """
            Apply the html_format void element to the text string.

            :params text: string to add the void element to
            :params end: place the void element at the end (True) or
                beginning (False)
            """
            if end is True:
                return f'{text}<{self.html_format}>'
            else:
                return f'<{self.html_format}>{text}'
        return void

    def wrap_txt_with_tag(self):
        """
        wraps the text string with the html tag
        """
        def tag(text):
            """
            Wrap the html_format element around the text string.

            :params text: string to add the void element to
            """
            return f'<{self.html_format}>{text}</{self.html_format}>'
        return tag

    def wrap_txt_with_tagclass(self):
        """
        wraps the text string with the html tag
        """
        def tagclass(text, html_class):
            """
            Wrap the html_format element around the text string and specify
            the html class.

            :params text: string to add the void element to
            :params html_class: specify the class for the specific html class
            """
            return f'<{self.html_format} class="{html_class}">' \
                   f'{text}</{self.html_format}>'
        return tagclass

    def wrap_txt_with_tagattr(self):
        """
        wraps the text string with the html tag
        """
        def tagattr(text, attr):
            """
            Wrap the html_format element around the text string and specify
            the attribute input.

            :params text: string to add the void element to
            :params attr: specify the html attributes for the specific html
            element
            """
            return f'<{self.html_format} {attr}>' \
                   f'{text}</{self.html_format}>'
        return tagattr

    @classmethod
    def href(cls, text, link):
        return cls('a').wrap_txt_with_tagattr()(
            text=text, attr=f'href="{link}" target="_blank"')

    @classmethod
    def div(cls):
        """
        Start a div class html block
        """
        return cls('div').wrap_txt_with_tagclass()

    @classmethod
    def linebr(cls):
        """
        insert a line break at the beginning and end of line
        """
        return cls('br').void_tag()

    @classmethod
    def bold(cls):
        """
        bold this text
        """
        return cls('b').wrap_txt_with_tag()

    @classmethod
    def par(cls):
        """
        Create a paragraph in html
        """
        return cls('p').wrap_txt_with_tag()

    @classmethod
    def lst(cls):
        """
        a bulleted list
        """
        return cls('li').wrap_txt_with_tag()

    @classmethod
    def indent_lst(cls, items, html_class=''):
        """
        create an indented bulleted item list

        :params items: a list of items you want to bullet
        :params html_class:
        """
        text = ''
        for item in items:
            text += cls('li').wrap_txt_with_tag()(text=item)
        return cls('ul').wrap_txt_with_tagclass()(html_class=html_class,
                                                  text=text)

    @classmethod
    def dlst(cls, items, html_class='dl-horizontal'):
        """
        Defines a description list.
        This is used with dt (bold list element) and dd (listed text under
        the bolded element).

        :params items: A nx2 nested list of strings to format.
        :params html_class: the class specifier for the dl html tag.
        """
        text = ''
        for item1, item2 in items:
            text += cls('dt').wrap_txt_with_tag()(text=item1) + \
                cls('dd').wrap_txt_with_tag()(text=item2)
        return cls('dl').wrap_txt_with_tagclass()(text=text,
                                                  html_class=html_class)

    @classmethod
    def dt(cls):
        """
        Defines a data term or name, for use with dlst.
        Formatting to bold headers in list elements under the dlst elements.
        """
        return cls('dt').wrap_txt_with_tag()

    @classmethod
    def dd(cls):
        """
        Defines the data definition, description, or value.
        Formatting for unbold text in list elements under the dlst tag.

        :params text: A string to wrap in hmtl dd format
        """
        return cls('dd').wrap_txt_with_tag()

    @staticmethod
    def normalize_string(text):
        """
        Replace new line characters with html line breaks for the text.
        """
        while text.endswith('\n'):
            text = text[:-1]
        while text.startswith('\n'):
            text = text[1:]
        while text.endswith('<br>'):
            text = text[:-len('<br>')]
        while text.startswith('<br>'):
            text = text[len('<br>'):]
        return text

    def combine_elements(self, items):
        """
        Combine a list of html elements with each element in the list on a new
        line with one line between them.
        """
        spacers = (self.html_format + self.html_format)
        items = [self.normalize_string(item) for item in items]

        return spacers.join(items)

    def get_recipe_href(self, asr_name, name=None):
        """
        Get a hyperlink for the recipe documentation associated with a given
        result.

        Parameters
        ----------
        asr_name : str
            asr_name variable of recipe
        name : str/None
            name for link - falls back to asr_name if None

        Returns
        -------
        link_name : str
        """
        if name is None:
            name = asr_name
        # ATM href only works to recipe main
        asr_name = asr_name.split('@')[0]
        link_name = self.href(text=name,
                              link="https://asr.readthedocs.io/en/"
                              f"latest/src/generated/recipe_{asr_name}.html")
        return link_name
