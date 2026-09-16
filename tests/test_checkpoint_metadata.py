import io
import pickle
import unittest

from scripts.inspect_checkpoint_metadata import MetadataReader, tensor_metadata


class CheckpointMetadataTests(unittest.TestCase):
    def test_untrusted_global_cannot_execute(self):
        class Untrusted:
            def __reduce__(self):
                return eval, ('1+2',)
        with self.assertRaises(pickle.UnpicklingError):
            MetadataReader(io.BytesIO(pickle.dumps(Untrusted()))).load()

    def test_no_external_storage_read_and_shape_preserved(self):
        reader = MetadataReader(io.BytesIO(b''))
        storage = reader.persistent_load(('storage', 'FloatStorage', '../../anything', 'cuda:0', 4096))
        meta = tensor_metadata(storage, 0, (32, 128), (128, 1))
        self.assertEqual(meta.shape, (32, 128))
        self.assertEqual(meta.dtype, 'FloatStorage')
        with self.assertRaises(pickle.UnpicklingError):
            reader.persistent_load(('file', '../../anything'))


if __name__ == '__main__':
    unittest.main()
