<img src=images/logo.png width=250px align=right>

# Deep Learning

This repo contains the material for GoDataDriven's deep learning course.


## Setting up 

*To create the anaconda environment*

* In the terminal/anaconda prompt, navigate to this directory

<!-- #region -->
* Install the virtual environment:

```bash
conda env create -f environment.yml
conda activate gdd-dl
```
<!-- #endregion -->

<!-- #region -->
*Alternatively you can use,*

```bash
pip install tensorflow
```
<!-- #endregion -->

## Structure

```
answers/                Python files with answers (loaded in Notebooks
                        with %load).
css/                    Presentation formatting.
data/                   Data.
exercises/              Exercises.
hackathon/              Hackathon.
images/                 Images for notebooks.
lectures/               Lectures.

```


## Syllabus 

### Day 1: Deep Learning introduction
- Introduction round
- Deep learning overview (EXERCISE: XOR thought experiment)
- Neural Network intution
- EXERCISE: XOR perceptron
- Deep learning basics
- Keras basics
- EXERCISE: Keras basics

### Day 2: Deep Learning for Images
- Recap
- Image recognition(EXERCISE: Convolutions)
- EXERCISE: CNN basics
- NN in practice
- Keras advanced
- EXERCISE: Keras advanced
- Netowrk architectures & transfer learning
- EXERCISE: Transfer learning
- EXERCISE: German Traffic Sign dataset exercise

### Day 3: Deep Learning for Sequential data & NLP
- Recap
- RNN lecture
- EXERCISE: Airline passenger forecasting
- LSTM lecture
- EXERCISE: Human activity recognition
- NLP
- EXERCISE: NLP sentiment classification


<!-- #region -->


### Hackathon options

There are two hackathon options. 
* hackathon/fashion-mnist-hackathon.ipynb: beginner
* hackathon/hackathon-pneumonia.ipynb: intermediate

The hackathon-pneumonia needs a large data set that is located [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download), which should be downloaded in advance.
<!-- #endregion -->

```python

```
