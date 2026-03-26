class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    @property
    def effective_length(self):
        return len(self)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        yield from range(len(self.data_source))
