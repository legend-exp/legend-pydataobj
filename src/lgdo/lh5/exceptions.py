from __future__ import annotations

import h5py


class LH5DecodeError(Exception):
    def __init__(
        self, message: str, file: str | h5py.File, oname: str | None = None
    ) -> None:
        super().__init__(message)

        self.file = file.filename if isinstance(file, h5py.File) else file
        self.obj = oname

    def __str__(self) -> str:
        if self.obj is None:
            msg = f"while opening file {self.file} for decoding: "
        else:
            msg = f"while decoding object '{self.obj}' in file {self.file}: "

        return msg + super().__str__()

    def __reduce__(self) -> tuple:  # for pickling.
        return self.__class__, (*self.args, self.file, self.obj)


class LH5EncodeError(Exception):
    def __init__(
        self,
        message: str,
        file: str | h5py.File,
        group: str | h5py.Group | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(message)

        self.file = file.filename if isinstance(file, h5py.File) else file
        self.group = (
            (group.name if isinstance(file, h5py.File) else group).rstrip("/")
            if group is not None
            else None
        )
        self.name = name.lstrip("/") if name is not None else None

    def __str__(self) -> str:
        if self.name is None:
            msg = f"while opening file {self.file} for encoding: "
        else:
            msg = (
                f"while encoding object {self.group}/{self.name} to file {self.file}: "
            )
        return msg + super().__str__()

    def __reduce__(self) -> tuple:  # for pickling.
        return self.__class__, (*self.args, self.file, self.group, self.name)
