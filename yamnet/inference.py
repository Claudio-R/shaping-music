# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Edited to be imported

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import numpy as np
import resampy
import soundfile as sf
import tensorflow as ts

from . import params as yamnet_params
from . import yamnet as yamnet_model

class YamNet:
    def __init__(self):
        self.params = yamnet_params.Params()
        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        self.yamnet.load_weights('yamnet_pkg/yamnet.h5')
        self.yamnet_classes = yamnet_model.class_names('yamnet_pkg/yamnet_class_map.csv')
        self.input = self.yamnet.input
  
    def __call__(self, audio):
        # Decode the WAV file.
        wav_data, sr = sf.read(audio, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != self.params.sample_rate:
            waveform = resampy.resample(waveform, sr, self.params.sample_rate)

        # Predict YAMNet classes.
        scores, embeddings, spectrogram = self.yamnet(waveform)
        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        labels = [self.yamnet_classes[i] for i in top5_i]
        return labels
    
    def summary(self):
        self.yamnet.summary()
    
    def get_layer(self, layer_name):
        return self.yamnet.get_layer(layer_name)
    