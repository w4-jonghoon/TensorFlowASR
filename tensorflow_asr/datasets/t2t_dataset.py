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
                 tfrecords_dirs: List[str],
                 speech_featurizer: speech_featurizers.SpeechFeaturizer,
                 text_featurizer: text_featurizers.TextFeaturizer,
                 stage: str,
                 data_paths: list = [],
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

    # DIFF(jseo): Remove decode wav and make label as int32
    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor,
                      indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = self.augmentations.signal_augment(audio)
            features = self.speech_featurizer.tf_extract(signal)
            features = self.augmentations.feature_augment(features)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            label_length = tf.cast(tf.shape(indices)[0], tf.int32)

            prediction = self.text_featurizer.prepand_blank(indices)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

            return (path, features, input_length, indices, label_length,
                    prediction, prediction_length)

    def create(self, batch_size: int):
        """Create tf.data.Dataset instance and preprocess data"""
        dataset = self._get_parsed_dataset(shuffle=self.shuffle)
        return self.process(dataset, batch_size)

    def _get_parsed_dataset(self, shuffle: bool = False):
        """From the list of files, get parsed dataset"""
        def _parse_example(examples):
            feature_description = {
                "wav_path":
                tf.io.FixedLenFeature([], tf.string),
                "waveforms":
                tf.io.FixedLenSequenceFeature((),
                                              tf.float32,
                                              allow_missing=True),
                TARGET_FIELD:
                tf.io.FixedLenSequenceFeature((), tf.int64,
                                              allow_missing=True),
            }
            example = tf.io.parse_single_example(examples, feature_description)
            example[TARGET_FIELD] = tf.cast(example[TARGET_FIELD], tf.int32)
            return example['wav_path'], example['waveforms'], example[
                TARGET_FIELD]

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False

        files = [
            os.path.join(tfrecord_dir, f'speech-*{self.stage}*')
            for tfrecord_dir in self.tfrecords_dirs
        ]

        # This data pipeline is based on:
        # https://stackoverflow.com/q/58014123/14091761

        return tf.data.Dataset.list_files(
            files, shuffle=shuffle,
            seed=0).with_options(ignore_order).interleave(
                tf.data.TFRecordDataset,
                cycle_length=tf.data.experimental.AUTOTUNE,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                    _parse_example).cache().prefetch(
                        tf.data.experimental.AUTOTUNE)

    def compute_metadata(self):
        """Compute metadata"""
        # pylint: disable=arguments-differ
        dataset = self._get_parsed_dataset(shuffle=False)
        max_input_length = 0
        max_target_length = 0
        for path, waveform, target in dataset:
            del path  # Unused
            if len(waveform) > max_input_length:
                max_input_length = len(waveform)
            if len(target) > max_target_length:
                max_target_length = len(target)

        # The below calculation is originally based on:
        #   SpeechFeaturizer.get_length_from_duration()
        if self.speech_featurizer.center:
            max_input_length += self.speech_featurizer.nfft
        max_input_length = 1 + (max_input_length - self.speech_featurizer.nfft
                                ) // self.speech_featurizer.frame_step
        self.speech_featurizer.update_length(max_input_length)
        self.text_featurizer.update_length(max_target_length)

        return super().compute_metadata()
