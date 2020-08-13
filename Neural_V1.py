"""
 Neural Network Intro
 2020-08-02
"""
 
import sys
import numpy as np
import random
import copy

#Input Layer
#First layer within the network responsible for taking inputs

class Layer_Input:
	#Forward Pass
	#We will be ignoring training in this first case as the network will be evolving genetically
	def forward(self, inputs):
		self.output = inputs


#Dense Layer
#Basic Layer Structure
class Layer_Dense:

	#Layer Initialization
	def __init__(self, n_inputs, n_neurons):
		#Initialize both weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

		#This initializes the biases to zero.  We may not want this as the genetic algorithm has no way to alter the biases through mating
		#self.biases = np.zeros((1, n_neurons))

		self.biases = 0.01 * np.random.randn(1, n_neurons)

	#Forward Pass Through Neuron
	def forward(self, inputs):
		#Calculate the basic output value of the neuron usings the dot product between weights and inputs and adding the bias
		self.output = np.dot(inputs, self.weights) + self.biases

#Rectified Linear Activaiton to be used within the hidden layers
class Activation_ReLu:
	#Forward Pass
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

#Softmax Activation class to be used for output neurons
class Activation_SoftMax:
	#Forward Pass
	def forward(self, inputs):
		#Get unnormalized probabilities
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

		#Normalize for each sample
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		self.output = probabilities

#Overall Model Class that will house all pieces of the network
class Model:

	def __init__(self):
		#Create a list of the objects in the network
		self.layers = []
		self.score = 0

	#Add objects to the model
	def add(self, layer):
		self.layers.append(layer)

	#In this initial case because the snake will be relying on a fitness function rather than a direct loss optimization we will be ignoring loss/optimizer settings

	#Finalize the model
	def finalize(self):

		#Create and set the input layer
		self.input_layer = Layer_Input()

		#Count all the objects within the model
		layer_count = len(self.layers)

		#Initialize a list of the layers we want to train
		self.trainable_layers = []

		#Iterate through the model objects
		for i in range(layer_count):
			#If it's the first layer, the previous object will be the input layer
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]

				#For all layers except the first and last
			elif i < layer_count - 1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]

			#Last layer
			#Again in this case we are ignoring loss so the last layer will be the output of the network
			else:
				self.layers[i].prev = self.layers[i-1]
				self.output_layer_activation = self.layers[i]

			#Check to see if the layer is trainable (in our case genetically modifiable)
			#Check for weights only is enough.  No need to look for biases
			if hasattr(self.layers[i], 'weights'):
				self.trainable_layers.append(self.layers[i])

				#We are ignoring the loss aspect of the training b/c genetic

	def mate(self, mom):
		
		child = copy.deepcopy(self)
		for i in range(len(child.layers)):

			if hasattr(child.layers[i], 'weights'):

				for x in range(self.layers[i].weights.shape[0]):
					for j in range(self.layers[i].weights.shape[1]):

						pass_on = bool(random.getrandbits(1))

						if pass_on == True:
							self.layers[i].weights[x][j] = self.layers[i].weights[x][j]
						else:
							self.layers[i].weights[x][j] = mom.layers[i].weights[x][j]

				for j in range(self.layers[i].biases.shape[1]):

					pass_on = bool(random.getrandbits(1))
					
					if pass_on == True:
						self.layers[i].biases[0][j] = self.layers[i].biases[0][j]
					else:
						self.layers[i].biases[0][j] = mom.layers[i].biases[0][j]
				'''
				if pass_on == True:
					child.layers[i].weights = self.layers[i].weights
					child.layers[i].biases = self.layers[i].biases
				else:
					child.layers[i].weights = mom.layers[i].weights
					child.layers[i].biases = mom.layers[i].biases
				'''


		child.score = 0

		return child

	#Perform the large forward pass of the model
	def forward(self, X):

		#Set the initial output of the first input layer of neurons y=X
		self.input_layer.forward(X)

		#Call the forward method of every object within the chain
		for layer in self.layers:
			layer.forward(layer.prev.output)

		# 'layer' is now the last object in the list so we should return its output
		#In our case this probably should be the probabilities of the choices
		return layer.output


def Model_Creator(self, n_inputs, n_neurons, n_outputs):

	#Instantiate the model
	model = Model()

	#Add layers
	model.add(Layer_Dense(n_inputs,n_neurons))
	model.add(Activation_ReLu())
	model.add(Layer_Dense(n_neurons,n_outputs))
	model.add(Activation_SoftMax())

	#Finalize the model
	model.finalize()

	return model

#Define a super generation class to include a population of models
class Generation:
	def __init__(self, n_inputs, n_neurons, n_outputs, mating = True,  population_size = 100, threshold = 10):
		
		self.population_size = population_size
		self.mating = mating
		self.threshold = threshold

		self.population = []
		for i in range(population_size):
			self.population.append(Model_Creator(self, n_inputs, n_neurons, n_outputs))

	def score_sort(self):

		self.population.sort(key = lambda x: x.score, reverse = True)

	def generation_mate(self):

		new_population = []

		for i in range(self.population_size):
			p1_indx = i % self.threshold
			p2_indx = min(self.population_size - 1, int(np.random.exponential(self.threshold)))

			offspring = self.population[p1_indx].mate(self.population[p2_indx])

			new_population.append(offspring)

		new_population[-1] = self.population[0]

		self.population = new_population



'''
#Basic model testing with Sentdex data
#Create the custom data set based on Sentdex course
X, y = spiral_data(100, 3)

#Instantiate the model
model = Model()

#Add layers
model.add(Layer_Dense(2,512))
model.add(Activation_ReLu())
model.add(Layer_Dense(512,3))
model.add(Activation_SoftMax())

#Finalize the model
model.finalize()

#This should be placed within a "training algorithm"
output = model.forward(X)

'''


'''

#This is all the hardcoded testing that was done originally

#Create a dense layer with inputs and outputs
dense1 = Layer_Dense(num_inputs, num_outputs)

dense2 = Layer_Dense(dense1.weights.shape[0] + 1, num_outputs)

print(dense1.weights.shape[0])

#Create activation object for hidden layers
activation1 = Activation_ReLu()
activation2 = Activation_SoftMax()

#Make a forward pass with our training data
dense1.forward(X)

#Run initial neuron layer outputs through ReLu activation
activation1.forward(dense1.output)

#Run outputs of activaiton1 through the second hidden layer
dense2.forward(activation1.output)

#Run the outputs of the second hidden layer through the SofMax activation function
activation2.forward(dense2.output)

#Look at the first few samples
print(activation2.output[:5])

'''


