class AnalysisFailure(RuntimeError):
    def __init__(self, message: str, partial_result):
        super().__init__(message)
        self.partial_result = partial_result
