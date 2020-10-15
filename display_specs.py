import librosa.display
import numpy as np
import matplotlib.pyplot as plt



gamma = np.load('/home/javier/repos/DCASE2020-Task3/input_feature/gammatone_gcclogmel/mic_dev/fold1_room1_mix001_ov1.npy')

gamma_ch1 = gamma[:,0:64]

plt.subplot(2, 2, 1)
gamma_ch1 = gamma_ch1.T
librosa.display.specshow(np.flip(gamma_ch1,1))
plt.colorbar()
plt.title('fold1_room1_mix001_ov1 gammatone scale to max')

gamma_norm = np.load('/home/javier/repos/DCASE2020-Task3/input_feature/gammatone_nomax_gcclogmel/mic_dev/fold1_room1_mix001_ov1.npy')

gamma_norm_ch1 = gamma_norm[:,0:64]
#gamma_norm_ch1 = gamma_norm_ch1.T
plt.subplot(2, 2, 2)
librosa.display.specshow(gamma_norm_ch1.T)
plt.colorbar()
plt.title('fold1_room1_mix001_ov1 gammatone no scale to max')

spec = np.load('/home/javier/repos/DCASE2020-Task3/input_feature/baseline_log_mel/mic_dev/fold1_room1_mix001_ov1.npy')

spec_ch1 = spec[:,0:64]

plt.subplot(2, 2, 3)
#spec_ch1 = spec_ch1.T
librosa.display.specshow(spec_ch1.T)
plt.colorbar()
plt.title('fold1_room1_mix001_ov1 mel spectrogram')

spec_norm = np.load('/home/javier/repos/DCASE2020-Task3/input_feature/baseline_log_mel/mic_dev_norm/fold1_room1_mix001_ov1.npy')

spec_norm_ch1 = spec_norm[:,0:64]

plt.subplot(2, 2, 4)
librosa.display.specshow(spec_norm_ch1.T)
plt.colorbar()
plt.title('fold1_room1_mix001_ov1 mel norm spectrogram')

plt.show()