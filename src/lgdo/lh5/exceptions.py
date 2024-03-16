from __future__ import annotations

import h5py


class LH5DecodeError(Exception):
    def __init__(self, message: str, file: str, obj: str) -> None:
        super().__init__(message)

        self.file = file.filename if isinstance(file, h5py.File) else file
        self.obj = obj

    def __str__(self) -> str:
        return (
            f"while reading object '{self.obj}' in file {self.file}: "
            + super().__str__()
        )
