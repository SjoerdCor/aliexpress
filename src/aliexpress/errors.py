class ReadableError(Exception):
    def __init__(self, code, context=None, technical_message=None):
        super().__init__(technical_message or code)
        self.code = code
        self.context = context or {}
        self.technical_message = technical_message


class ValidationError(ReadableError):
    pass


class FeasibilityError(ReadableError):
    pass


class CouldNotReadFileError(ReadableError):
    pass
