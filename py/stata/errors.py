class StataError(Exception):
    """Error carrying a Stata return code."""

    def __init__(self, msg, rc=198):
        super().__init__(msg)
        self.msg = msg
        self.rc = rc


class ExitDo(Exception):
    """`exit` inside a do-file."""


class BreakLoop(Exception):
    pass
