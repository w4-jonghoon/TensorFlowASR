"""Unit tests for ASRDatasets"""

import glob
import os
import shutil
import tempfile
import unittest

from tensorflow_asr.datasets import t2t_dataset
from tensorflow_asr.featurizers import speech_featurizers
from tensorflow_asr.featurizers import text_featurizers

TESTDATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'testdata'))


class ASRT2TDatasetTests(unittest.TestCase):
    """Unit tests for ASRT2TDataset class"""
    def setUp(self):
        pass

    def test_create(self):
        """Test for create() function"""
        dataset = t2t_dataset.ASRT2TDataset(
            data_paths=[],
            tfrecords_dirs=[TESTDATA_DIR],
            speech_featurizer=speech_featurizers.TFSpeechFeaturizer(
                speech_config={}),
            text_featurizer=text_featurizers.CharFeaturizer(decoder_config={}),
            stage='train',
            indefinite=True,
            use_tf=True,
        )
        data_loader = dataset.create(1)

        actual_count = 0
        for data in data_loader:
            del data  # Unused
            actual_count += 1
        expected_count = 1198

        self.assertEqual(expected_count, actual_count)

        # Multi dir case
        src_tfrecord = glob.glob(os.path.join(TESTDATA_DIR, 'speech-*'))[0]
        with tempfile.TemporaryDirectory() as dir_0:
            with tempfile.TemporaryDirectory() as dir_1:
                shutil.copy(src_tfrecord, dir_0)
                shutil.copy(src_tfrecord, dir_1)

                dataset = t2t_dataset.ASRT2TDataset(
                    data_paths=[],
                    tfrecords_dirs=[dir_0, dir_1],
                    speech_featurizer=speech_featurizers.TFSpeechFeaturizer(
                        speech_config={}),
                    text_featurizer=text_featurizers.CharFeaturizer(
                        decoder_config={}),
                    stage='train',
                    indefinite=True,
                    use_tf=True,
                )
                data_loader = dataset.create(1)

                actual_count = 0
                for data in data_loader:
                    del data  # Unused
                    actual_count += 1
        expected_count = 1198 * 2

        self.assertEqual(expected_count, actual_count)


if __name__ == '__main__':
    unittest.main()
