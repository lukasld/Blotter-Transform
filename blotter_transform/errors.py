class Error(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class RequiredColumnsNotPresentError(Error):
    """
    Description of the Error
    """

    def __init__(self, expected_names, actual_names):
        self.message = f"provided column names: {actual_names}, do not match required names: {expected_names}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
