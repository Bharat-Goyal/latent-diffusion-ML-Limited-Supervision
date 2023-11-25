import numpy as np
from pesq import pesq
from ldm.data.voicebank import VoiceBankTest
from os import listdir
from preprocess import make_spectrum
from einops import rearrange, repeat
import librosa
import scipy
from ldm.util import default
from ldm.models.diffusion.ddim import DDIMSampler

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/voicebank-ldm-kl-3.yaml")  
    model = load_model_from_config(config, "/home/hice1/skamath36/scratch/latent-diffusion-ML-Limited-Supervision/logs/2023-11-23T14-21-00_voicebank-ldm-kl-3/checkpoints/epoch=000004.ckpt")
    return model


if __name__ == '__main__':
    with torch.no_grad():
        noisy_dir = "/home/hice1/skamath36/scratch/noisy_testset_wav/noisy_testset_wav/"
        clean_dir = "/home/hice1/skamath36/scratch/clean_testset_wav/"
        model = get_model()
        sampler = DDIMSampler(model)
        clean_wav_files = [f for f in listdir(clean_dir)]
        noisy_wav_files = [f for f in listdir(noisy_dir)]
        assert len(clean_wav_files) == len(noisy_wav_files)
        pesq_sum = 0
        pesq_good_files, pesq_good_values = [], []
        for i in range(len(clean_wav_files)):
            # Get and format spectrogram for testing
            spec, phase, len_y = make_spectrum(noisy_dir + noisy_wav_files[i], FRAMELENGTH=1024, SHIFT=256)
            spec_tensor = torch.tensor(spec.astype(np.float32))[:,:243]
            if spec_tensor.size()[1] < 243:
                m = torch.nn.ZeroPad2d((0,243-spec_tensor.size()[1],0,0))
                spec_tensor = m(spec_tensor)
            spec_tensor = spec_tensor.unsqueeze(2).unsqueeze(0)
            spec_tensor = spec_tensor.to("cuda:0")
            spec_tensor = rearrange(spec_tensor, 'b h w c -> b c h w')
            spec_tensor = spec_tensor.to(memory_format=torch.contiguous_format).float()

            # Pass spectrogram through first stage
            model_output = model.first_stage_model(spec_tensor)
            z = model.get_first_stage_encoding(model_output[1])
            # Pass through forward and sampling stages
            loss, loss_dict, model_output = model.forward(z, c=None)

            # Decode
            x_pred = model.first_stage_model.decode(model_output)

            # Crop output to original size
            x_pred = x_pred.squeeze(0).squeeze(0).cpu().detach().numpy()
            crop = min(243, spec.shape[1])
            x_pred = x_pred[:,:crop]

            # Invert STFT
            D_new = phase[:,:crop] * np.expm1(x_pred)
            pred_wav = librosa.istft(D_new, hop_length=256, win_length=1024, n_fft=1024, window=scipy.signal.hamming)
            clean_wav, sr = librosa.load(clean_dir + clean_wav_files[i], sr=16000)
            pesq_score = pesq(sr, clean_wav[:pred_wav.size], pred_wav)
            pesq_sum += pesq_score
            if (i % 100 == 0):
                print("Iteration f{%d}",i)
                print(pesq_score)
            if pesq_score >= 2:
                pesq_good_files.append(noisy_wav_files[i])
                pesq_good_values.append(pesq_score)
        print(pesq_sum)
        print(pesq_good_values)
        print(pesq_good_files)
        print(pesq_sum / len(clean_wav_files))

