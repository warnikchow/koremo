# Koremo
5-class Korean emotion clssifier

## Requirements
librosa, Keras (TensorFlow), Numpy

## Simple usage
<pre><code> from koremo import pred_emo(filename) </code></pre>
* file in .wav format is recommended

## Data preperation
Voice recorded by two Korean voice actors (1 male, 1 female)
### Emotion categorization
* Angry (Female 1000, Male 800)
* Fear (Female 500, Male 550)
* Joy (Female 1000, Male 1000)
* Normal (Female 2700, Male 2699)
* Sad (Female 500, Male 800)
### The dataset was primarily constructed for the following paper:
```
@article{lee2018acoustic,
  title={Acoustic Modeling Using Adversarially Trained Variational Recurrent Neural Network for Speech Synthesis},
  author={Lee, Joun Yeop and Cheon, Sung Jun and Choi, Byoung Jin and Kim, Nam Soo and Song, Eunwoo},
  journal={Proc. Interspeech 2018},
  pages={917--921},
  year={2018}
}
```
* Cite the article for EITHER the reference of the classification criteria or the usage of the toolkit.

## System architecture
* The model adopts a concatenated structure of CNN and BiLSTM Self-attention, as in [Korinto](https://github.com/warnikchow/korinto), and the only change is the third convolutional layer window (3 by 3 >> 5 by 5)
* The best model shows accuracy: 96.45% and F1: 0.9597, wit train:test set ratio 9:1.


