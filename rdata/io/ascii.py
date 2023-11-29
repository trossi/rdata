from __future__ import annotations

import string
import numpy as np


from typing import (
    TextIO,
)

from rdata.parser._parser import (
    AltRepConstructorMap,
    DEFAULT_ALTREP_MAP,
    Parser,
    RData,
)

from .base import Writer


R_INT_NA = 'NA' # noqa: WPS432
"""Value used to represent a missing integer in R."""


class ParserASCII(Parser):
    """Parser for files in ASCII format."""

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


class WriterASCII(Writer):
    """Writer for files in ASCII format."""

    def __init__(
        self,
        file: TextIO,
    ) -> None:
        self.file = file

    def _writeline(self, line) -> None:
        """Write a line with trailing \\n"""
        self.file.write(f'{line}\n')

    def write_magic(self):
        self._writeline('A')

    def write_nullable_bool(self, value):
        if value is None or np.ma.is_masked(value):
            self._writeline('NA')
        else:
            self.write_int(int(value))

    def write_nullable_int(self, value):
        if value is None or np.ma.is_masked(value):
            self._writeline('NA')
        else:
            self._writeline(value)

    def write_double(self, value):
        self._writeline(str(value))

    def write_string(self, value: bytes):
        self.write_int(len(value))

        # This line would produce byte representation in hex such as '\xc3\xa4':
        # value = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we use a somewhat manual conversion instead:
        value = ''.join(chr(byte) if chr(byte) in string.printable else rf'\{byte:03o}' for byte in value)

        self._writeline(value)
