import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from settings import *

class conv2d_bn(tf.keras.layers.Layer):
    def __init__(self, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None, bias=False):
        super(conv2d_bn, self).__init__()
        self.activation = activation
        self.conv2D = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=bias)
        self.batchNorm = BatchNormalization(axis=3, scale=False)
        self.activation_layer = Activation(activation=activation, name=name)

    def call(self, input):
        """
        Builds Conv2D blocks
        """
        x = self.conv2D(input)
        x = self.batchNorm(x)

        if(self.activation == None):
            return x

        x = self.activation_layer(x)

        return x

class MultiResBlock(tf.keras.layers.Layer):
    def __init__(self, U, alpha = ALPHA):
        super(MultiResBlock, self).__init__()
        self.W = alpha * U
        self.shortcut = conv2d_bn(int(self.W*0.167) + int(self.W*0.333) +
                            int(self.W*0.5), 1, 1, activation=None, padding='same')
        self.conv3x3 = conv2d_bn(int(self.W*0.167), 3, 3,
                            activation='relu', padding='same')
        self.conv5x5 = conv2d_bn(int(self.W*0.333), 3, 3,
                            activation='relu', padding='same')
        self.conv7x7 = conv2d_bn(int(self.W*0.5), 3, 3,
                            activation='relu', padding='same')

        self.batchNorm = BatchNormalization(axis=3)
        self.activation = Activation('relu')
        self.batchNorm2 = BatchNormalization(axis=3)

    def call(self, input):
        """
        Builds MultiRes blocks
        """
        shortcut = input

        shortcut = self.shortcut(shortcut)
        conv3x3 = self.conv3x3(input)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)

        out = concatenate([conv3x3, conv5x5, conv7x7])
        out = self.batchNorm(out)

        out = add([shortcut, out])
        out = self.activation(out)
        out = self.batchNorm2(out)

        return out

class ResPath(tf.keras.layers.Layer):
    def __init__(self, filters, length):
        super(ResPath, self).__init__()
        self.length = length
        self.shortcut = conv2d_bn(filters, 1, 1,
                            activation=None, padding='same')
        self.out = conv2d_bn(filters, 3, 3, activation='relu', padding='same')

        self.activation = Activation('relu')
        self.batchNorm = BatchNormalization(axis=3)
        self.length_shortcut = [conv2d_bn(filters, 1, 1, activation=None, padding='same'),
            conv2d_bn(filters, 1, 1, activation=None, padding='same'),
            conv2d_bn(filters, 1, 1, activation=None, padding='same'),
            conv2d_bn(filters, 1, 1, activation=None, padding='same')]
        self.length_out = [conv2d_bn(filters, 3, 3, activation='relu', padding='same'),
            conv2d_bn(filters, 3, 3, activation='relu', padding='same'),
            conv2d_bn(filters, 3, 3, activation='relu', padding='same'),
            conv2d_bn(filters, 3, 3, activation='relu', padding='same')]
        self.length_activation = [Activation('relu'),
            Activation('relu'),
            Activation('relu'),
            Activation('relu')]
        self.length_batchNorm = [BatchNormalization(axis=3),
            BatchNormalization(axis=3),
            BatchNormalization(axis=3),
            BatchNormalization(axis=3)]

    def call(self, input):
        """
        Builds Res Paths
        """
        shortcut = input
        shortcut = self.shortcut(shortcut)
        out = self.out(input)
        out = add([shortcut, out])
        out = self.activation(out)
        out = self.batchNorm(out)

        for i in range(self.length-1):

            shortcut = out
            shortcut = self.length_shortcut[i](shortcut)
            out = self.length_out[i](out)
            out = add([shortcut, out])
            out = self.length_activation[i](out)
            out = self.length_batchNorm[i](out)

        return out

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, n_coeff):
        super(AttentionBlock, self).__init__()
        self.w_gate = conv2d_bn(n_coeff, 1, 1, activation=None, bias=True)
        self.w_x = conv2d_bn(n_coeff, 1, 1, activation=None, bias=False)
        self.psi = conv2d_bn(n_coeff, 1, 1, activation='softmax', bias=True)
        self.convTransp = Conv2DTranspose(n_coeff, (2, 2), strides=(2, 2), padding='same')
        self.activation = Activation('relu')
        self.w = conv2d_bn(n_coeff, 1, 1, activation=None, bias=True)
        self.w2 = conv2d_bn(n_coeff, 1, 1, activation='relu', bias=True)

    def call(self, res_path, input):
        """
        Builds Attention blocks
        """
        g = self.w_gate(input)
        x = self.w_x(res_path)
        x = tf.add(self.convTransp(g), x)
        psi = self.activation(x)
        psi = self.psi(psi)
        out = tf.multiply(res_path, psi)
        out = self.w(out)
        out = self.w2(out)
        return out, psi

class Attentive_MultiResUNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.filters = 32
        self.a1 = MultiResBlock(self.filters)
        self.a2 = ResPath(self.filters, 4)
        self.a3 = MaxPooling2D(pool_size=(2, 2), padding='same')

        self.a4 = MultiResBlock(self.filters*2)
        self.a5 = ResPath(self.filters*2, 3)
        self.a6 = MaxPooling2D(pool_size=(2, 2), padding='same')

        self.a7 = MultiResBlock(self.filters*4)
        self.a8 = ResPath(self.filters*4, 2)
        self.a9 = MaxPooling2D(pool_size=(2, 2), padding='same')

        self.a10 = MultiResBlock(self.filters*8)
        self.a11 = ResPath(self.filters*8, 1)
        self.a12 = MaxPooling2D(pool_size=(2, 2), padding='same')

        self.a13 = MultiResBlock(self.filters*16)
        self.a14 = conv2d_bn(self.filters*8, 1, 1, activation='relu', padding='same')

        self.a15 = AttentionBlock(self.filters*8)
        self.a16 = Conv2DTranspose(self.filters*8, (2, 2), strides=(2, 2), padding='same')
        self.a17 = MultiResBlock(self.filters*8)

        self.a18 = AttentionBlock(self.filters*4)
        self.a19 = Conv2DTranspose(self.filters*4, (2, 2), strides=(2, 2), padding='same')
        self.a20 = MultiResBlock(self.filters*4)

        self.a21 = AttentionBlock(self.filters*2)
        self.a22 = Conv2DTranspose(self.filters*2, (2, 2), strides=(2, 2), padding='same')
        self.a23 = MultiResBlock(self.filters*2)

        self.a24 = Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='same')
        self.a25 = MultiResBlock(self.filters)


        self.predictions = Conv2D(filters=1, kernel_size=1, padding='same')

    def call(self, input):
        """
        Builds Attentive MultiResUNet model
        """
        x = input
        x = self.a1(x)
        res_1 = self.a2(x)
        x = self.a3(x)

        x = self.a4(x)
        res_2 = self.a5(x)
        x = self.a6(x)

        x = self.a7(x)
        res_3 = self.a8(x)
        x = self.a9(x)

        x = self.a10(x)
        res_4 = self.a11(x)
        x = self.a12(x)

        x = self.a13(x)
        gating = self.a14(x)

        out, attention = self.a15(res_4, gating)
        x = self.a16(x)
        x = concatenate([out, x])
        x = self.a17(x)

        out, attention = self.a18(res_3, x)
        x = self.a19(x)
        x = concatenate([out, x])
        x = self.a20(x)

        out, attention = self.a21(res_2, x)
        x = self.a22(x)
        x = concatenate([out, x])
        x = self.a23(x)

        x = self.a24(x)
        x = concatenate([res_1, x])
        x = self.a25(x)

        predictions = self.predictions(x)
        return predictions
