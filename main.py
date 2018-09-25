import tensorflow as tf
import numpy as np
import librosa
import csv
from sympy.utilities.iterables import multiset_permutations
from glob import glob
from model import SEN
from pydub import AudioSegment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Batch():
    def __init__(self):
        self.x1 = []
        self.x2 = []
        self.y = []

def read_audio(f):
    avgv = np.load('model/avg.npy')
    stdv = np.load('model/std.npy')
    y, _ = librosa.core.load(f, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    S = np.transpose(np.log(1+10000*S))
    S = (S-avgv)/stdv
    S = np.expand_dims(S, 2)
    return S

def playMusicPuzzle(save_txt=True, save_csv=True, save_mp3=True):
    # all target audio
    fs = sorted(glob('data/*.mp3'))

    # extract mel-spectrogram and name
    target_name, target_fea = [], []
    for f in fs:
        S = read_audio(f)
        target_name.append(f.split('/')[-1][:-4])
        target_fea.append(S)

    # overall possible permutations (brute-force method)
    n = len(fs)
    ps = [p for p in multiset_permutations(np.arange(n))]
    ps = np.array(ps)

    # calculate pairwise-similarity
    with tf.Session() as sess:
        model = SEN(is_train=False)
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, 'model/model')
        score = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    batch = Batch()
                    batch.x1 = [target_fea[i]]
                    batch.x2 = [target_fea[j]]
                    batch.y = [[0, 0]]
                    score[i, j] = model.calculate(sess, batch)[0][1]

    # find the best permutation
    output = []
    for p in ps:
        temp = 0
        for z in range(len(p)-1):
            temp += score[p[z], p[z+1]]
        output.append(temp)
    best_p = ps[np.argmax(output)]

    # save best permutation (txt)
    if save_txt:
        with open('output/best_permutation.txt', mode='w', encoding='utf-8') as file:
            temp = [target_name[index] for index in best_p]
            file.write('%s' % ('\t'.join(temp)))

    # save pair-wise similarity (csv)
    if save_csv:
        with open('output/output.csv', mode='w', encoding='utf_8_sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + target_name)
            for n, s in zip(target_name, score):
                temp = [str(ss) for ss in s]
                temp = [n] + temp
                writer.writerow(temp)

    # save concatenated audio based on the best permutation (mp3)
    if save_mp3:
        result = ''
        for index in best_p:
            audio = AudioSegment.from_file(fs[index], format='mp3')
            if result == '':
                result = audio
            else:
                result += audio
            result.export('output/output.mp3', format='mp3')

if __name__ == '__main__':
    playMusicPuzzle(save_txt=True, save_csv=True, save_mp3=True)