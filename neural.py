#####################################################################################################################
## Title: Disply
#####################################################################################################################
## Description: Contains all the components of a modular genetic neural network with the following structure:
##              Generation --> Model --> Layers
##				Each generation can contain an arbitrary number of unique neural networks (models), which in turn
##				are built from any number of layers (input, hidden, and output).  Generations are evolved by
##				mating the models within them to pass along the best attributes of the previous generation
#####################################################################################################################
## Author: Alex Wadey
## Version: 1.0
## Email: wadeyalex@gmail.com
## Status: active
#####################################################################################################################
 
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
		#Initialize both weights and biases to some random small number
		self.weights = 1 * np.random.randn(n_inputs, n_neurons)
		self.biases = 1 * np.random.randn(1, n_neurons)

	#Forward Pass Through Neuron
	def forward(self, inputs):
		#Calculate the basic output value of the neuron usings the dot product between weights and inputs and adding the bias
		self.output = np.dot(inputs, self.weights) + self.biases

#Last layer used for obtaining output of model
class Layer_Output:

	#Layer Initialization
	def __init__(self, n_inputs, n_outputs):

		#There are no biases for the output layer
		self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)

	def forward(self, inputs):
		#Calculate forward pass through the output layer
		self.output = np.dot(inputs, self.weights)


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

	def __init__(self, n_inputs, n_neurons, n_outputs):
		#Create a list of the objects in the network
		self.layers = []
		self.score = 0
		self.n_inputs = n_inputs
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs

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

	#mating function to propogate "good genes" to the next generation
	def mate(self, mom, mutate = False, mutate_strength = 0.05):
		
		#create a copy of the model that was passed into the function
		child = copy.deepcopy(self)

		#For the models we are mating and not mutating
		if mutate == False:

			#For all the layers within the model
			for i in range(len(child.layers)):
				#If the layer is a weight layer (i.e. a neuron layer and not an activation layer)
				if hasattr(child.layers[i], 'weights'):

					#Iterate over all weights within the layer and pass on either of the parent's weights depending on the random bool pass_on
					for x in range(child.layers[i].weights.shape[0]):
						for j in range(child.layers[i].weights.shape[1]):

							pass_on = bool(random.getrandbits(1))

							if pass_on == True:
								child.layers[i].weights[x][j] = self.layers[i].weights[x][j]
							else:
								child.layers[i].weights[x][j] = mom.layers[i].weights[x][j]
					
				#If the layer is a bias layer (redundant: if a layer has weights it will typically have biases)
				if hasattr(child.layers[i], "biases"):

					#iterate over biases and do the same thing as above
					for j in range(child.layers[i].biases.shape[1]):

						pass_on = bool(random.getrandbits(1))
						
						if pass_on == True:
							child.layers[i].biases[0][j] = self.layers[i].biases[0][j]

						else:
							child.layers[i].biases[0][j] = mom.layers[i].biases[0][j]
		
		#If we are mutating instead of mating come here	
		else:
			
			#For each layer within the model
			for i in range(len(child.layers)):

				#If the layer has weights
				if hasattr(child.layers[i], 'weights'):

					#Iterate through each of the weights and mutate them
					for x in range(child.layers[i].weights.shape[0]):
						for j in range(child.layers[i].weights.shape[1]):
							#Randomly decide if the weight should be tweaked more positively or negatively
							up_down = random.uniform(-mutate_strength, mutate_strength)
							#Mutate the weight
							child.layers[i].weights[x][j]*= (1+up_down)

				#If layer has biases
				if hasattr(child.layers[i], 'biases'):
					#Iterate through each bias and mutate
					for j in range(child.layers[i].biases.shape[1]):
						#Randomly decide if the weight should be tweaked more positively or negatively
						up_down = random.uniform(-mutate_strength, mutate_strength)
						#Mutate the weight
						child.layers[i].biases[0][j]*= (1+up_down)

		#Set the score of the new model to zero
		child.score = 0
         
        #Return the new model
		return child

	#Perform the large forward pass of the model
	def forward(self, X):

		#Set the initial output of the first input layer of neurons y=X
		self.input_layer.forward(X)

		#Call the forward method of every object within the chain
		for layer in self.layers:
			layer.forward(layer.prev.output)

		# 'layer' is now the last object in the list so we should return its output
		#In our case this will be the probabilities of the choices
		return layer.output

	#Write a model to a text file
	def Model_Write(self):

		#Open text file with write permissions
		f = open("Model_Weights.txt", "w+")

		#Initially write the key details of the model
		f.write("%s " % self.n_inputs)
		f.write("%s " % self.n_neurons)
		f.write("%s " % self.n_outputs)

		#Iterate across the layers in the model
		for i in range(len(self.layers)):
			#If the layer has weights
			if hasattr(self.layers[i], 'weights'):

				#New line to divide between weighted layers
				f.write("\n")

				for x in range(self.layers[i].weights.shape[0]):
					
					#New line to divide between weight sets
					f.write("\n")

					for j in range(self.layers[i].weights.shape[1]):

						#Write model weights to file
						f.write("%s " % self.layers[i].weights[x][j])

		#Close file
		f.close

		#Open model biases text file
		f = open("Model_Biases.txt", "w+")

		#Initially write the key details of the model
		f.write("%s " % self.n_inputs)
		f.write("%s " % self.n_neurons)
		f.write("%s " % self.n_outputs)

		#for each layer in the model
		for i in range(len(self.layers)):

			#If layer has bias
			if hasattr(self.layers[i], 'biases'):

				#Add new line in between bias sets
				f.write("\n")
				f.write("\n")

				for j in range(self.layers[i].biases.shape[1]):
					#Write biases to file
					f.write("%s " % self.layers[i].biases[0][j])

		#Close file
		f.close

	#Read model weights and biases from file
	def Model_Read(self):

		#Try to open file, but if not possible return error message
		try:
			f = open("Model_Weights.txt", "r")

		except:
			print("unable to open stored model data")
			return

		#Set iterable defaults
		first = True
		i = -1
		x = 0

		#For each line in the file
		for line in f:

			#Split the line into a list of words
			values = line.split()

			#If the line is empty alter iterables
			if values == []:
				x = -1
				i+=1

			#If line is not empty
			else:

				#If this is the first line then check the model details to see if they match
				if first == True:
					#Set first to false after reading first line
					first = False

					#Check to see if model parameters match those in the written files
					if int(values[0]) != self.n_inputs:
						print("Number of Model Inputs Does Not Match")
						return -1

					elif int(values[1]) != self.n_neurons:
						print("Number of Model Neurons Does Not Match")
						return -1

					elif int(values[2]) != self.n_outputs:
						print("Number of Model Outputs Does Not Match")
						return -1

				#If everything matches up
				else:
					#Iterate until we reach a layer that has weights
					while not(hasattr(self.layers[i], "weights")):
						i+=1

					#Iterate through the layer and add the values from file
					for j in range(self.layers[i].weights.shape[1]):

						self.layers[i].weights[x][j] = float(values[j])
			#Iterate
			x+=1

		#Close file
		f.close

		#Open biases file.  The assumption is made that if the weights file opened then the biases file will to
		f = open("Model_Biases.txt", "r")

		#Set iterables and boolean
		first = True
		i = -1

		#For each line in the file
		for line in f:

			#Split the line into a list of words
			values = line.split()

			#If the line is empty alter iterables
			if values == []:
				i+=1

			else:

				if first == True:

					first = False

					#Check to see if model parameters match those in the written files
					if int(values[0]) != self.n_inputs:
						print("Number of Model Inputs Does Not Match")
						return -1

					elif int(values[1]) != self.n_neurons:
						print("Number of Model Neurons Does Not Match")
						return -1

					elif int(values[2]) != self.n_outputs:
						print("Number of Model Outputs Does Not Match")
						return -1

				else:
					
					#Iterate until a bias layer has been reached
					while not(hasattr(self.layers[i], "biases")):
						i+=1

					#Read the biases for that layer into the model
					for j in range(self.layers[i].biases.shape[1]):

						self.layers[i].biases[0][j] = float(values[j])


#Define a super generation class to include a population of models
class Generation:
	def __init__(self, n_inputs, n_neurons, n_outputs, mating = True,  population_size = 100, threshold = 10, mutate_threshold = 0.50):
		
		#Number of models per generation
		self.population_size = population_size
		#Boolean value to decide if mating is turned on or not
		self.mating = mating
		#Percentage of models to use in mating for the next generation
		self.threshold = threshold
		#The fraction of models under which models mate with one another.  i.e. 0.75 means 75% of models mate and 15% of models mutate off of the best model
		self.mutate_threshold = mutate_threshold

		#Create the model population
		self.population = []
		for i in range(population_size):
			#Create models with the predefined number of inputs, neurons, and outpus
			self.population.append(Model_Creator(self, n_inputs, n_neurons, n_outputs))

	#Sort the generation's population based on score
	def score_sort(self):
		#Sort the population based on score so highest becomes the first index
		self.population.sort(key = lambda x: x.score, reverse = True)
		#Return the top score of the generation
		return self.population[0].score

	#Mate models within the generation to produce the next generation offspring
	def generation_mate(self):

		#Define place holder for next generation
		new_population = []

		#Iterate over the population
		for i in range(self.population_size):
			#Loop through the self.threshold number of top models and use as parent one
			p1_indx = i % self.threshold

			#Choose parent two randomly from the population
			p2_indx = min(self.population_size - 1, int(np.random.exponential(self.threshold)))

			#If the model number is under the mutation threshold then mate without mutating
			if i < int(self.mutate_threshold*self.population_size):

				offspring = self.population[p1_indx].mate(self.population[p2_indx])

			#If the model number is above the mutation threshold mate with mutating
			else:

				offspring = self.population[p1_indx].mate(self.population[p2_indx], mutate= True)
			#Add the offspring to the new population
			new_population.append(offspring)

		#In order to preserve the best models pass along the best model from the previous generation to the new one
		new_population[-1] = self.population[0]

		#Overwrite the old model population with the new one
		self.population = new_population

def Model_Creator(self, n_inputs, n_neurons, n_outputs):

	#Instantiate the model
	model = Model(n_inputs, n_neurons, n_outputs)

	#Add layers
	model.add(Layer_Dense(n_inputs,n_neurons))
	model.add(Activation_ReLu())
	model.add(Layer_Dense(n_neurons,n_neurons))
	model.add(Activation_ReLu())
	model.add(Layer_Output(n_neurons,n_outputs))
	model.add(Activation_SoftMax())

	#Finalize the model
	model.finalize()

	return model