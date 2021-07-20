"""Generate metadata from Skelter Labs' T2T TFRecords"""

import argparse
import os

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets import t2t_dataset
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers
from tensorflow_asr.featurizers import text_featurizers
from tensorflow_asr.utils.file_util import preprocess_paths

parser = argparse.ArgumentParser(prog="Generate meta files from T2T TFRecords")

parser.add_argument("--stage",
                    type=str,
                    default="train",
                    help="The stage of dataset")

parser.add_argument("--config",
                    type=str,
                    default=None,
                    help="The file path of model configuration file")

parser.add_argument("--metadata",
                    type=str,
                    default=None,
                    help="Path to file containing metadata")

parser.add_argument("tfrecord_dirs",
                    nargs="+",
                    type=str,
                    default=None,
                    help="Paths to tfrecord files")
args = parser.parse_args()

assert args.metadata is not None, "metadata must be defined"

tfrecord_dirs = preprocess_paths(args.tfrecord_dirs)

config = Config(args.config)

speech_featurizer = speech_featurizers.TFSpeechFeaturizer(
    speech_config=config.speech_config)

# Set text featurizer as default one. Because it is no affect to meta data.
text_featurizer = text_featurizers.CharFeaturizer(decoder_config={})

dataset = t2t_dataset.ASRT2TDataset(
    tfrecords_dirs=tfrecord_dirs,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    stage=args.stage,
    shuffle=False,
)

dataset.update_metadata(args.metadata)
