import io
import os
import tarfile
import zipfile
from typing import Optional
import urllib.request
from urllib.parse import urlparse

import tqdm


class ProgressIO(io.FileIO):
    def __init__(self, file, **kwargs):
        super().__init__(file, mode="rb")

        kwargs["total"] = kwargs.get("total", None)

        if (kwargs["total"] is None) and self.seekable():
            init_pos = self.tell()
            self.seek(0, os.SEEK_END)
            kwargs["total"] = self.tell()
            self.seek(init_pos, os.SEEK_SET)

        self.process = tqdm.tqdm(**kwargs)

    def read(self, __size: int = -1) -> bytes | None:
        self.process.update(__size)

        return super().read(__size)


def uncompress_progress(
    file: str,
    member: str,
    path: Optional[str] = "",
    **kwargs,
):
    if tarfile.is_tarfile(file):
        with tarfile.open(
            fileobj=ProgressIO(
                file,
                unit="Mo",
                unit_scale=True,
                miniters=1 << 20,
                leave=False,
                **kwargs,
            )
        ) as f:
            f.extract(member, path=path, filter="data")

    elif zipfile.is_zipfile(file):
        with zipfile.ZipFile(
            ProgressIO(file, unit="Mo", unit_scale=True, miniters=1 << 20, **kwargs)
        ) as f:
            f.extract(member, path=path)

    else:
        raise Exception(f"file could not be opened")


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_filename(url: str) -> str:
    return os.path.split(urlparse(url).path)[1]


def download_progress(url: str, output: Optional[str] = "./", **kwargs):
    path, filename = os.path.split(output)

    path = os.path.abspath(path)
    if len(filename) == 0:
        filename = get_filename(url)

    with DownloadProgressBar(
        unit="Mo", unit_scale=True, miniters=1 << 20, leave=False, **kwargs
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=os.path.join(path, filename),
            reporthook=t.update_to,
        )
