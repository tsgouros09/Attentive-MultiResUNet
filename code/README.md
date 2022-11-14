
# An efficient Short-Time Discrete Cosine Transform and Attentive MultiResUNet framework for Music Source Separation

## Abstract
The music source separation problem, where the task at hand is to estimate the audio components that are present in a mixture, has been at the centre of research activity for a long time. In more recent frameworks, the problem is tackled by creating deep learning models, which attempt to extract information from each component by using {Short-Time Fourier Transform (STFT)} spectrograms as input. {Most approaches assume that one source is present at each time-frequency point, which allows to allocate this point from the mixture to the desired source. Since this assumption is strong and is reported not to hold in practice, there is a problem that arises from the use of the magnitude of the STFT as input to these networks, which is the absence of the Fourier phase information during the separated source reconstruction.}  The recovery of the Fourier phase information is neither easily tractable, nor computationally efficient to estimate. In this paper, we propose a novel Attentive MultiResUNet architecture, that uses real-valued Short-Time Discrete Cosine Transform data as inputs. This step avoids the phase recovery problem, by estimating the appropriate values within the network itself, rather than employing complex estimation or post-processing algorithms. The proposed novel network features a U-Net type structure with residual skip connections and an attention mechanism that correlates the skip connection and the decoder output at the previous level. The proposed network is used for the first time in source separation and is more computationally efficient than state-of-the-art separation networks and features favourable performance compared to the state-of-the-art with a fraction of the computational cost.

![Attentive MultiResUNet](AttentMultiResUNet.jpg)

## Paper

The paper is available from [here](https://).

## Instructions

1. In this folder, you can find the python code for Attentive MultiResUNet. This model requires Tensorflow v 2.8.3.

2. Before training a model, open the settings.py file and change the path for MUSDB18 to where it is located in your drive.
You can also change the parameters in settings.py file to your preference.

3. To train a new model, run the training.py file

4. After training, the new model will be saved in the Models folder. Therefore, before evaluation, in the settings.py file, change the path for the trained model.

## Audio Samples

Audio samples can be found [here](https://tsgouros09.github.io/Attentive-MultiResUNet). 

## Citation
Please cite the following paper, if you are using this code.

T. Sgouros, A. Bousis, N. Mitianoudis, "An efficient Short-Time Discrete Cosine Transform and Attentive MultiResUNet framework for Music Source Separation", IEEE Access, Vol. , 2022.
