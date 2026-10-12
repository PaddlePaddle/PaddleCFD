class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        keys = ", ".join(sorted(self.__dict__.keys()))
        return f"Data({keys})"
