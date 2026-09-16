"""Inspect PyTorch zip checkpoint tensor shapes without loading tensor bytes.

Restricted unpickling: only inert storage descriptors, tensor metadata, and
OrderedDict are accepted; checkpoint globals are never imported or executed.
This is a provenance/dimension audit, not a substitute for strict model loading.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import io
import json
from pathlib import Path
import pickle
import zipfile


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: str


def tensor_metadata(storage, storage_offset, size, stride, *unused):
    if not isinstance(storage, dict) or not isinstance(size, tuple):
        raise ValueError('Unknown tensor serialization')
    if len(size) > 16 or any(type(x) is not int or x < 0 for x in size):
        raise ValueError('Invalid tensor dimensions')
    return TensorMetadata(size, storage['dtype'])


def parameter_metadata(data, *unused):
    if not isinstance(data, TensorMetadata):
        raise ValueError('Unknown parameter serialization')
    return data


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if module == 'torch' and name == 'Size':
            return tuple
        if module in ('builtins', '__builtin__') and name == 'set':
            return set
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
            return tensor_metadata
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return parameter_metadata
        if module == 'torch' and name in ('FloatStorage', 'HalfStorage', 'BFloat16Storage',
                                         'DoubleStorage', 'LongStorage', 'IntStorage',
                                         'ShortStorage', 'CharStorage', 'ByteStorage', 'BoolStorage'):
            return name
        raise pickle.UnpicklingError(f'Disallowed checkpoint global {module}.{name}')

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != 'storage':
            raise pickle.UnpicklingError('Unknown persistent storage reference')
        return {'dtype': pid[1], 'elements': pid[4]}


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        candidates = [x for x in archive.infolist() if x.filename.endswith('/data.pkl')]
        if len(candidates) != 1 or candidates[0].file_size > 32 * 1024 * 1024:
            raise ValueError('Expected one bounded PyTorch data.pkl')
        value = MetadataReader(io.BytesIO(archive.read(candidates[0]))).load()
    if not isinstance(value, dict) or not isinstance(value.get('module'), dict):
        raise ValueError('Expected checkpoint[module] state dict')
    tensors = {}
    for key, tensor in value['module'].items():
        if not isinstance(key, str) or not isinstance(tensor, TensorMetadata):
            raise ValueError('Non-tensor state dict item')
        tensors[key] = {'shape': list(tensor.shape), 'dtype': tensor.dtype}
    return {'file': path.name, 'tensor_count': len(tensors), 'tensors': tensors,
            'strict_model_loading_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'file': result['file'], 'tensor_count': result['tensor_count'],
                      'strict_model_loading_verified': False}))


if __name__ == '__main__':
    main()
