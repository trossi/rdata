from __future__ import annotations

import io

from typing import (
    Any,
    BinaryIO,
)

import numpy as np
import numpy.typing as npt

from rdata.parser._parser import (
    AltRepConstructorMap,
    DEFAULT_ALTREP_MAP,
    Parser,
    RData,
)

from .base import Writer


R_INT_NA = -2**31  # noqa: WPS432
"""Value used to represent a missing integer in R."""


class ParserXDR(Parser):
    """Parser for files in XDR format."""

    def __init__(
        self,
        file: BinaryIO,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.file = file

    def _parse_array(
            self,
            dtype: np.dtype,
    ) -> npt.NDArray[Any]:  # noqa: D102
        length = self.parse_int()
        return self._parse_array_values(dtype, length)

    def _parse_array_values(
            self,
            dtype: np.dtype,
            length: int,
    ) -> npt.NDArray[Any]:  # noqa: D102
        dtype = np.dtype(dtype)
        buffer = self.file.read(length * dtype.itemsize)
        # Read in big-endian order and convert to native byte order
        return np.frombuffer(buffer, dtype=dtype.newbyteorder('>')).astype(dtype, copy=False)

    def parse_nullable_int(self) -> int | None:  # noqa: D102
        value = int(self._parse_array_values(np.int32, 1)[0])
        if value == R_INT_NA:
            return None
        else:
            return value

    def parse_double(self) -> float:  # noqa: D102
        return float(self._parse_array_values(np.float64, 1)[0])

    def parse_complex(self) -> complex:  # noqa: D102
        return complex(self._parse_array_values(np.complex128, 1)[0])

    def parse_nullable_int_array(
        self,
        fill_value: int = 0,
    ) -> npt.NDArray[np.int32] | np.ma.MaskedArray[Any, Any]:  # noqa: D102

        data = self._parse_array(np.int32)
        mask = data == R_INT_NA
        data[mask] = fill_value

        if np.any(mask):
            return np.ma.array(  # type: ignore
                data=data,
                mask=mask,
                fill_value=fill_value,
            )

        return data

    def parse_double_array(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._parse_array(np.float64)

    def parse_complex_array(self) -> npt.NDArray[np.complex128]:  # noqa: D102
        return self._parse_array(np.complex128)

    def parse_string(self, length: int) -> bytes:  # noqa: D102
        return self.file.read(length)

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        # Check that there is no more data in the file
        assert self.file.read(1) == b''
        return rdata


def flatten_nullable_int_array(array):
    assert array.dtype in (np.int32, int)
    if np.ma.is_masked(array):
        mask = np.ma.getmask(array)
        array = np.ma.getdata(array).copy()
        array[mask] = R_INT_NA
    return array


class WriterXDR(Writer):
    """Writer for files in XDR format."""

    def __init__(
        self,
        file: io.BytesIO,
    ) -> None:
        self.file = file

    def write_magic(self):
        self.file.write(b'X\n')

    def __write_array(self, array):
        # Expect only 1D arrays here
        assert array.ndim == 1
        self.write_int(array.size)
        self.__write_array_values(array)

    def __write_array_values(self, array):
        # Convert to big endian if needed
        array = array.astype(array.dtype.newbyteorder('>'))
        # 1D array should be both C and F contiguous
        assert array.flags['C_CONTIGUOUS'] == array.flags['F_CONTIGUOUS']
        if array.flags['C_CONTIGUOUS']:
            data = array.data
        else:
            data = array.tobytes()
        self.file.write(data)

    def write_nullable_bool(self, value):
        if value is None or np.ma.is_masked(value):
            value = R_INT_NA
        self.__write_array_values(np.array(value).astype(np.int32))

    def write_nullable_int(self, value):
        if value is None or np.ma.is_masked(value):
            value = R_INT_NA
        self.__write_array_values(np.array(value).astype(np.int32))

    def write_double(self, value):
        self.__write_array_values(np.array(value))

    def write_complex(self, value):
        self.__write_array_values(np.array(value))

    def write_nullable_bool_array(self, array):
        self.write_nullable_int_array(array.astype(np.int32))

    def write_nullable_int_array(self, array):
        array = flatten_nullable_int_array(array)
        self.__write_array(array.astype(np.int32))

    def write_double_array(self, array):
        self.__write_array(array)

    def write_complex_array(self, array):
        self.__write_array(array)

    def write_string(self, value: bytes):
        self.write_int(len(value))
        self.file.write(value)
