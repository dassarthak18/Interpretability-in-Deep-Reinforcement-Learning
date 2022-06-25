# Interpretability-in-Deep-Reinforcement-Learning
The Jupyter notebook was written for and run in Google Colaboratory, with the source code mounted in Google Drive. To run the notebook locally, Python 3.5+ as well as python3-pip is required. The following prerequisites must also be installed (tested in an Ubuntu 21.10 system).

## Installing OpenAI Gym

A Python toolkit for testing RL algorithms that provides an implementation for different standard RL environments. 

```shell
  $ sudo apt install build-essential python3-dev swig \
  python3-pygame git libosmesa6-dev libgl1-mesa-glx libglfw3
  $ sudo pip3 install ale-py atari-py AutoROM.accept-rom-license \
  lz4 opencv-python pyvirtualdisplay pyglet importlib-resources \
  Cython cffi glfw imageio lockfile pycparser pillow zipp gym
```

## Installing Keras-RL2

A deep RL library for Keras that has implementations of state-of-the-art RL algorithms such as DQN, CEM and SARSA. Integrates with OpenAI Gym out of the box, and has support for TensorFlow 2. 

```shell
  $ git clone https://github.com/wau/keras-rl2.git
  $ cd keras-rl2
  $ sudo pip3 install .
```

## Installing SciKit-Learn

A Python machine learning library that features various classification, regression and clustering algorithms including decision tree classifiers. 

```shell
  $ sudo pip3 install sklearn
```

## Other Prerequisites

* **Graphviz** - A Python toolkit for drawing graphs specified in DOT language scripts. 
* **Matplotlib** - A Python plotting library, specifically for NumPy.
* **Pandas** - A Python toolkit for data manipulation and analysis.
* **tqdm** - A Python library to enable iterables to show a smart progress meter.

```shell
  $ sudo pip3 install graphviz matplotlib pandas tqdm
  $ sudo apt install graphviz
```
