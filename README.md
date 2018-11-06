# Koremo
5-class Korean emotion classifier

## Requirements
librosa, Keras (TensorFlow), Numpy

## Simple usage
<pre><code> from koremo import pred_emo(filename) </code></pre>
* file in *.wav* format is recommended
* Output in five labels **0: Angry**, **1: Fear**, **2: Joy**, **3: Normal**, **4: Sad**
* ONLY ACOUSTIC DATA is utilized

## Data preperation
Voice recorded by two Korean voice actors (1 male, 1 female)
### Categorizing emotions
* Angry (Female: 1,000 / Male: 800)
* Fear (Female: 500 / Male: 550)
* Joy (Female: 1,000 / Male: 1,000)
* Normal (Female: 2,700 / Male: 2,699)
* Sad (Female: 500 / Male: 800)
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
* Cite the [ARTICLE](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1598.pdf) for EITHER the reference of the classification criteria and the concept of acoustic feature-based Korean emotion classification. Note that the source *.wav* files are not disclosed currently.
* Also, cite THIS repository for the usage of the toolkit.
```
@article{cho2018koremo,
  title={KorEmo: 5-class Korean emotion classifier},
  author={Cho, Won Ik},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/warnikchow/koremo}
  year={2018}
}
```

## System architecture
* The model adopts a concatenated structure of CNN and BiLSTM Self-attention, as in [Korinto](https://github.com/warnikchow/korinto), and the only change is the third convolutional layer window (3 by 3 >> 5 by 5)
* The best model shows **accuracy: 96.45%** and **F1: 0.9597**, with train:test set ratio 9:1.


