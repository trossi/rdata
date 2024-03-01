import io
import pytest

import rdata
import rdata.io


TESTDATA_PATH = rdata.TESTDATA_PATH


def decompress_data(data):
    from rdata.parser._parser import file_type, FileTypes

    filetype = file_type(data)

    if filetype is FileTypes.bzip2:
        from bz2 import decompress
    elif filetype is FileTypes.gzip:
        from gzip import decompress
    elif filetype is FileTypes.xz:
        from xz import decompress
    else:
        decompress = lambda x: x

    return decompress(data)


fnames = sorted([fpath.name for fpath in TESTDATA_PATH.glob("*.rd?")])

@pytest.mark.parametrize("fname", fnames, ids=fnames)
def test_write(fname):
    with (TESTDATA_PATH / fname).open("rb") as f:
        data = decompress_data(f.read())
        rds = data[:2] != b'RD'
        format = 'ascii' if data.isascii() else 'xdr'

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        if format == 'ascii':
            fd = io.StringIO()
            data = data.decode('ascii')
            data = data.replace('\r\n', '\n')
        else:
            fd = io.BytesIO()

        try:
            rdata.io.write_file(fd, r_data, format=format, rds=rds)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        out_data = fd.getvalue()

        if data != out_data:
            print(r_data)
            print('in')
            print(data)
            print('out')
            print(out_data)

        assert data == out_data


@pytest.mark.parametrize("fname", fnames, ids=fnames)
def test_convert_to_r(fname):
    with (TESTDATA_PATH / fname).open("rb") as f:
        # Skip test files without unique R->py->R transformation
        if fname in [
            "test_encodings.rda",     # encoding not kept in Python
            "test_encodings_v3.rda",  # encoding not kept in Python
            "test_list_attrs.rda",    # attributes not kept in Python
            "test_file.rda",          # attributes not kept in Python
        ]:
            pytest.skip("ambiguous R->py->R transformation")

        data = decompress_data(f.read())
        rds = data[:2] != b'RD'
        format = 'ascii' if data.isascii() else 'xdr'

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        try:
            py_data = rdata.conversion.convert(r_data)
        except NotImplementedError as e:
            pytest.skip(str(e))

        encoding = r_data.extra.encoding
        if encoding is None:
            if "win" in fname:
                encoding = "CP1252"
            else:
                encoding = "UTF-8"

        try:
            new_r_data = rdata.conversion.convert_to_r_data(py_data, rds=rds, versions=r_data.versions, encoding=encoding)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        if r_data != new_r_data:
            print('ref')
            print(r_data)
            print('py')
            print(py_data)
            print('new')
            print(new_r_data)

        assert r_data == new_r_data
