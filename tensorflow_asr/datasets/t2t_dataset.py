"""ASRDataset for Skelter Labs' T2T TFRecords"""

import os
from typing import List

import tensorflow as tf

from tensorflow_asr.augmentations import augmentation
from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.datasets.base_dataset import AUTOTUNE, BUFFER_SIZE
from tensorflow_asr.featurizers import speech_featurizers
from tensorflow_asr.featurizers import text_featurizers

TARGET_FIELD = 'targets/tts_jamo2'


class ASRT2TDataset(asr_dataset.ASRDataset):
    """Dataset for ASR using T2T TFRecords"""
    def __init__(self,
                 data_paths: list,
                 tfrecords_dirs: List[str],
                 speech_featurizer: speech_featurizers.SpeechFeaturizer,
                 text_featurizer: text_featurizers.TextFeaturizer,
                 stage: str,
                 augmentations: augmentation.Augmentation = (
                     augmentation.Augmentation(None)),
                 cache: bool = False,
                 shuffle: bool = False,
                 use_tf: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        # pylint: disable=too-many-arguments
        super().__init__(stage=stage,
                         speech_featurizer=speech_featurizer,
                         text_featurizer=text_featurizer,
                         data_paths=data_paths,
                         augmentations=augmentations,
                         cache=cache,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         drop_remainder=drop_remainder,
                         use_tf=use_tf,
                         indefinite=indefinite)
        if not self.stage:
            raise ValueError(
                "stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dirs = tfrecords_dirs

    # DIFF(jseo): Remove decode wav
    def preprocess(self, path: tf.Tensor, audio: tf.Tensor,
                   indices: tf.Tensor):
        with tf.device("/CPU:0"):

            def fn(_path: bytes, _audio: bytes, _indices: bytes):
                signal = self.augmentations.signal_augment(_audio)
                features = self.speech_featurizer.extract(signal.numpy())
                features = self.augmentations.feature_augment(features)
                features = tf.convert_to_tensor(features, tf.float32)
                input_length = tf.cast(tf.shape(features)[0], tf.int32)

                label = tf.strings.to_number(tf.strings.split(_indices),
                                             out_type=tf.int32)
                label_length = tf.cast(tf.shape(label)[0], tf.int32)

                prediction = self.text_featurizer.prepand_blank(label)
                prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

                return _path, features, input_length, label, label_length, prediction, prediction_length

            return tf.numpy_function(fn,
                                     inp=[path, audio, indices],
                                     Tout=[
                                         tf.string, tf.float32, tf.int32,
                                         tf.int32, tf.int32, tf.int32, tf.int32
                                     ])

    # DIFF(jseo): Remove decode wav
    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor,
                      indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = self.augmentations.signal_augment(audio)
            features = self.speech_featurizer.tf_extract(signal)
            features = self.augmentations.feature_augment(features)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            label = tf.cast(indices, tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)

            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

            return path, features, input_length, label, label_length, prediction, prediction_length

    def parse(self, record: tf.Tensor):
        """Parse and preprocess a TFRecord example

        This method only performs parsing TFRecord example, but
        preprocessing is performed in parent's method.

        Args:
            record: contains an example protobuf of each TFRecord record.

        Returns:
            (inputs, labels)
            The first element (inputs) is a dict which has keys:
                inputs, inputs_length, predictions, predictions_length
            The second element (labels) is a dict which has keys:
                labels, labels_length
        """
        # pylint: disable=arguments-differ
        feature_description = {
            "wav_path": tf.io.FixedLenFeature([], tf.string),
            "waveforms": tf.io.VarLenFeature(tf.float32),
            TARGET_FIELD: tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(record, feature_description)
        # Re-map to TensorFlowASR keys
        example = {
            "path": example["wav_path"],
            "audio": tf.sparse.to_dense(example['waveforms']),
            "indices": tf.sparse.to_dense(example[TARGET_FIELD]),
        }
        return super().parse(**example)

    def create(self, batch_size: int):
        """Create tf.data.Dataset instance and preprocess data"""
        patterns = [
            os.path.join(tfrecord_dir, f'speech-*{self.stage}*')
            for tfrecord_dir in self.tfrecords_dirs
        ]
        files_ds = tf.data.Dataset.list_files(patterns,
                                              shuffle=self.shuffle,
                                              seed=0)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds,
                                          num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)
