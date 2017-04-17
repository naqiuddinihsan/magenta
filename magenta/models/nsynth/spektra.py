import utils as mgu
import numpy as np
from scipy.io.wavfile import read
import math

# test 1
a = read('/Users/ihsan/GitHub/mfcc/speech.wav')

# native loadwav
# a = mgu.load_wav('/Users/ihsan/GitHub/mfcc/speech.wav')

# convert audio file into np.array?
audio = np.array(a[1], dtype=float)

# preparam (from MATLAB)
frameLen = 25
frameShiftMS = 10

# parameter setup
mag = mgu.specgram(audio)  # y
phase_angle = 0
n_fft = pow(2, (math.ceil(math.log(frameLen)/math.log(2))))
hop = frameShiftMS
num_iters = 50

# griffin_lim function
mgu.griffin_lim(mag, phase_angle, n_fft, hop, num_iters)
