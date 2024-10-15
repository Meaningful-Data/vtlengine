import csv


class RFCDialect(csv.Dialect):
    """
    https://docs.python.org/3/library/csv.html#csv.Dialect
    https://tools.ietf.org/html/rfc4180
    """

    delimiter = ","
    doublequote = True
    lineterminator = "\r\n"
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
    strict = True
    escapechar = None
    skipinitialspace = False


def register_rfc() -> None:
    """Register the RFC dialect."""
    csv.register_dialect("rfc", RFCDialect)
