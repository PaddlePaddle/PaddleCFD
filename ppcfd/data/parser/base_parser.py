# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Type, Union


class DataParserMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if hasattr(cls, "file_format"):
            formats = cls.file_format if isinstance(cls.file_format, list) else [cls.file_format]
            for fmt in formats:
                DataParserFactory.register_loader(fmt)(cls)


class DataParserFactory:
    _loaders: Dict[str, Type] = {}

    @classmethod
    def register_loader(cls, format_name: str):
        def decorator(loader_class: Type):
            cls._loaders[format_name.lower()] = loader_class
            return loader_class

        return decorator

    @classmethod
    def get_loader(cls, format_name: str, **kwargs) -> object:
        loader_class = cls._loaders.get(format_name.lower())
        if not loader_class:
            available = ", ".join(cls._loaders.keys())
            raise ValueError(f"No loader for {format_name} now. Available: {available}")
        return loader_class(**kwargs) if loader_class else None


class BaseTransition(metaclass=DataParserMeta):
    def __init__(self, file_path: Union[str, Path]):
        """Basic transition class.

        Args:
            file_path (Union[str, Path]): path of data.
        """
        super().__init__()
        self.file_path = Path(file_path).absolute()

    @abstractmethod
    def get_data(self):
        """Must be defined in the subclass to ensure that the loader can obtain the data."""
        pass
