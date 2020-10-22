"""Module for registering dependencies between recipes."""


class DependencyStack:
    """Maintain a stack like object for dependecies."""

    def __init__(self):
        self.stack = []

    def register_results(self, resultid):
        for frame in self.stack:
            frame.append(resultid)

    def add_frame(self):
        frame = []
        self.stack.append(frame)
        return frame

    def pop_frame(self):
        return self.stack.pop()

    def __enter__(self):
        """Add frame to dependency stack."""
        frame = self.add_frame()
        return frame

    def __exit__(self):
        """Pop frame off of dependency stack."""
        self.pop_frame()


dependency_stack = DependencyStack()
