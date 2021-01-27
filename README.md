# Evolutionary Neural Network Playing Snake (Snake_AI)

### Project Description

The aim of this project was to create a neural network from scratch in Python and have it learn the game Snake.
To achieve this a evolutionary neural network structure was decided upon in which generations of neural networks are compete for the best scores.
The best performing models from each generation are then "bred" to pass along their best characteristics

The project consists of four main files:

- main.py
  - contains main project loop and instantiates primary class structures for neural network
- neural.py
  - contains all components necessary to build evoluationary neural network (applicablation independent)
- game_logic.py
  - contains game logic code to track the status of the Snake game and check for game over states
- display.py
  - contains pygame details to present the game and the neural network progress to the user

## Getting Started

### Prerequisites

This project requires the following libraries, easily installed using pip

- copy
- numpy
- random
- pygame

### Installation

1. Clone the repo
`git clone https://github.com/ooshka/Snake_AI.git`

2. Use pip to install required libraries
`pip install pygame`

## Usage

The primary use case for this project is training evolutionary neural networks.  This is done by simply running the main function loop; however, the user can decide on a number of parameters for the neural network that can alter the learning process significantly.

### main.py

Within main.py the following parameters are editable by the user (future commits will have these values directly inputted by user):

`pop size` - The model population size of each generation (how many neural networks per generation)\
`trials` - The number of trials each model is given to play Snake before the next model takes its turn\
`generations` - The number of generations to be tested.  Each generation will be bred from the one before it\
`n_neurons` - The number of neurons to be used for each hidden layer of the neural network (more takes more comp time but may lead to faster learning)\

### display.py

The initialization for the Display class contains all parameters governing the spacing and look of the pygame GUI.  If desired the user can alter these values to change the spacing and shape of the game display

## Contact

Alex Wadey - wadeyalex@gmail.com

Project Link - https://github.com/ooshka/Snake_AI


