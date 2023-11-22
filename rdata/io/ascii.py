from __future__ import annotations

from typing import (
    TextIO,
)

from rdata.parser._parser import (
    AltRepConstructorMap,
    DEFAULT_ALTREP_MAP,
    Parser,
    RData,
)


R_INT_NA = 'NA' # noqa: WPS432
"""Value used to represent a missing integer in R."""


class ParserASCII(Parser):
    """Parser used when the file is in ASCII format."""

    def __init__(
        self,
        file: TextIO,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.file = file

    def _readline(self) -> str:
        """Read a line without trailing \\n"""
        return self.file.readline()[:-1]

    def parse_nullable_int(self) -> int | None:  # noqa: D102
        value = self._readline()
        if value == R_INT_NA:
            return None
        else:
            return int(value)

    def parse_double(self) -> float:  # noqa: D102
        return float(self._readline())

    def parse_string(self, length: int) -> bytes:  # noqa: D102
        return self._readline().encode('ascii').decode('unicode_escape').encode('latin1')

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        # Check that there is no more data in the file
        assert self.file.read(1) == ''
        return rdata


