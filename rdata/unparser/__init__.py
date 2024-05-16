"""Utilities for unparsing a rdata file."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os

    from rdata.parser import RData

    from ._ascii import UnparserASCII
    from ._xdr import UnparserXDR


def unparse_file(
        path: os.PathLike[Any] | str,
        r_data: RData,
        *,
        file_format: str = "xdr",
        rds: bool = True,
        compression: str = "gzip",
) -> None:
    """
    Unparse RData object to a file.

    Parameters
    ----------
    path:
        File path to be created
    r_data:
        RData object
    file_format:
        File format (ascii or xdr)
    rds:
        Whether to create RDS or RDA file
    compression:
        Compression (gzip, bzip2, xz, or none)
    """
    if compression == "none":
        from builtins import open  # noqa: UP029
    elif compression == "bzip2":
        from bz2 import open  # type: ignore [no-redef]
    elif compression == "gzip":
        from gzip import open  # type: ignore [no-redef]
    elif compression == "xz":
        from lzma import open  # type: ignore [no-redef]
    else:
        msg = f"Unknown compression: {compression}"
        if compression is None:
            msg += ". Use 'none' for no compression."
        raise ValueError(msg)

    with open(path, "wb") as f:
        unparse_data(f, r_data, file_format=file_format, rds=rds)


def unparse_data(
        fileobj: IO[Any],
        r_data: RData,
        *,
        file_format: str = "xdr",
        rds: bool = True,
) -> None:
    """
    Unparse RData object to a file object.

    Parameters
    ----------
    fileobj:
        File object
    r_data:
        RData object
    file_format:
        File format (ascii or xdr)
    """
    Unparser: type[UnparserXDR | UnparserASCII]  # noqa: N806

    if file_format == "ascii":
        from ._ascii import UnparserASCII as Unparser
    elif file_format == "xdr":
        from ._xdr import UnparserXDR as Unparser
    else:
        msg = f"Unknown file format: {file_format}"
        raise ValueError(msg)

    unparser = Unparser(fileobj)  # type: ignore [arg-type]
    unparser.unparse_r_data(r_data, rds=rds)
