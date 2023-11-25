# Copyright 2020 LMNT, Inc. All Rights Reserved.
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

import numpy as np
import os
import random
import torch
import librosa

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class VoiceBankTrain(torch.utils.data.Dataset):
  def __init__(self, npy_path, crop, voicebank=False):
    super().__init__()
    # self.filenames = []
    self.specnames = []
    # self.se = se
    self.voicebank = voicebank
    self.crop = crop
    print(npy_path)
    self.specnames = glob(f'{npy_path}/*.wav.spec.npy', recursive=True)

  def __len__(self):
    return len(self.specnames)

  def __getitem__(self, idx):
    spec_filename = self.specnames[idx]                
    spectrogram = np.load(spec_filename)
    spec_tensor = torch.tensor(spectrogram.astype(np.float64))[:,:self.crop]
    if spec_tensor.size()[1] < self.crop:
      m = torch.nn.ZeroPad2d((0,self.crop-spec_tensor.size()[1],0,0))
      spec_tensor = m(spec_tensor)
    return {"image" : spec_tensor.unsqueeze(2)}
       
class VoiceBankValidation(torch.utils.data.Dataset):
  def __init__(self, npy_path, crop, voicebank=False):
    super().__init__()
    # self.filenames = []
    self.specnames = []
    # self.se = se
    self.voicebank = voicebank
    self.crop = crop
    self.specnames = glob(f'{npy_path}/*.wav.spec.npy', recursive=True)

  def __len__(self):
    return len(self.specnames)

  def __getitem__(self, idx):
    spec_filename = self.specnames[idx]                
    spectrogram = np.load(spec_filename)
    spec_tensor = torch.tensor(spectrogram.astype(np.float64))[:,:self.crop]
    if spec_tensor.size()[1] < self.crop:
      m = torch.nn.ZeroPad2d((0,self.crop-spec_tensor.size()[1],0,0))
      spec_tensor = m(spec_tensor)
    return {"image" : spec_tensor.unsqueeze(2)}
class VoiceBankTest(torch.utils.data.Dataset):
  def __init__(self, clean_wav_path, noisy_wav_path, voicebank=False):
    super().__init__()
    # self.filenames = []
    self.sound_names_clean = []
    self.sound_names_noisy = []
    # self.se = se
    self.voicebank = voicebank
    self.sound_names_clean = glob(f'{clean_way_path}/*.wav', recursive=True)
    self.sound_names_noisy = glob(f'{noisy_wav_path}/*.wav', recursive=True)

  def __len__(self):
    return len(self.specnames)

  def __getitem__(self, idx):
    sound_clean_file = self.sound_names_clean[idx]
    sound_noisy_file = self.sound_names_noisy[idx]
                
    # spectrogram = np.load(spec_filename)
    # spec_tensor = torch.tensor(spectrogram).unsqueeze(2)
    return sound_clean_file, sound_noisy_file

if __name__ == "__main__":
  trainset = VoiceBankTrain("/home/hice1/skamath36/scratch/latent-diffusion-ML-Limited-Supervision/spec/voicebank_Clean/train", 81)
  x = 0
  print(trainset[x].size())
  print(x)