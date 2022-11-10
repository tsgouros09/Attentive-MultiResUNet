import numpy as np
import tensorflow as tf
import scipy

def frameWind(x, frame, hop_size):
    """
    Returns STDCT of signal x (with Tensorflow)
    x :
    """
    M = len(x)
    K = np.int(np.floor(M / (frame * hop_size)))
    W = tf.signal.hamming_window(frame)
    length = K-np.int(np.floor(1 / hop_size - 1))
    X = []
    index = 0
    for i in range(length):
        X.append(tf.signal.dct(x[index:index + frame] * W, norm='ortho'))
        index += np.int(np.floor(frame * hop_size))
    X = tf.stack(X, axis=-1)
    return X

def inverseWind(X, hop_size):
    """
    Returns iSTDCT of signal X (with Tensorflow)
    X :
    """
    frame, K = X.shape
    M = K + np.int(np.floor(1 / hop_size - 1))
    y = []
    index = 0
    for i in range(K):
        W = tf.signal.hamming_window(frame)
        ytmp = tf.signal.idct(X[:,i], norm='ortho')*W
        ytmp = tf.pad(ytmp,[[0, np.int(frame * M * hop_size) - frame]])
        y.append(tf.roll(ytmp, shift=index, axis=0))
        index += np.int(np.floor(frame * hop_size))
    y = tf.reduce_sum(y, axis=0)
    return y

def frameWind_eval(x, frame, hop_size):
    """
    Returns STDCT of signal x (with scipy)
    x :
    """
    M = len(x)
    K = np.int(np.floor(M / (frame * hop_size)))
    W = np.hamming(frame)
    length = K-np.int(np.floor(1 / hop_size)) - 1
    X = []
    index = 0
    for i in range(length):
        X.append(scipy.fft.dct(x[index:index + frame] * W, norm='ortho'))
        index += np.int(np.floor(frame * hop_size))
    X = np.stack(X, axis=-1)
    return X

def inverseWind_eval(X, hop_size):
    """
    Returns iSTDCT of signal X (with scipy)
    X :
    """
    frame, K = X.shape
    M = K + np.int(np.floor(1 / hop_size)) - 1
    y = np.zeros(int(np.ceil(frame * M * hop_size)))
    index = 0
    for i in range(K-1):
        W = np.hamming(frame)
        ytmp = scipy.fft.idct(X[:,i], norm='ortho')*W
        y[index:index + frame] += ytmp
        index += np.int(np.floor(frame * hop_size))
    return y
