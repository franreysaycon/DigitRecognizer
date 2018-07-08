from PIL import Image, ImageTk
import numpy as np
from numpy import *
from random import *
from functools import reduce
from math import *
import operator
import os, sys
import re
import pickle, cloud
import pylab as pl
import copy

seed(12345)
class pf(float):

	def __repr__(self):
		return "%-3.4f" % self

class Network:

	def __init__(self, layers=[]):

		self.layers = []

		for n in range(1,len(layers)):
			n_in = layers[n-1]
			n_nodes = layers[n]

			name = 'h%d'%n if n+1 < len(layers) else 'o'
			l = Layer(n_nodes, n_in, name=name) 
			self.layers.append(l)


	def feed_forward(self, inputs): 
	
		new_inp=list(inputs)
		
		for l in self.layers:
			new_inp = l.feed_forward(new_inp)	
			l.outputs = list(new_inp)

		return new_inp


	def back_propagate(self, target, lr=0.3, mtm=0.6):

		for l in range(len(self.layers)-1,-1,-1):
			out_errors=[]
			if l == len(self.layers)-1: 
				for i in range(0,len(target)):
					t_o = target[i] - self.layers[l].outputs[i]
					out_errors.append(t_o)
				self.layers[l].back_propagate(out_errors, lr, mtm)
				
			else:
				w=0
				for n in range(0,self.layers[l].n_nodes):
					sum_err = 0
					w = n+1
					for m in range(len(self.layers[l+1].errors)):
						sum_err = sum_err + (self.layers[l+1].weights[w]*self.layers[l+1].errors[m])
						w = w + self.layers[l+1].n_in+1
					out_errors.append(sum_err)
				self.layers[l].back_propagate(out_errors, lr, mtm)

		return None

	def classify(self, inputs, outputs):

		actual = self.feed_forward(inputs)
		
		classified = [0]*len(actual)
		for i in range(len(classified)):
			if actual[i] > 0.5: classified[i] = 1 
		
		return classified

	def __repr__(self):
		s = "ANN with following weights:\n"
		for l in self.layers:
			s += l.__repr__() + "\n"
		return s	

class Layer:

	def __init__(self, n_nodes, n_in, f_act='sigmoid', name='l', b=1):
		self.name = name

		self.b = b 
		self.n_nodes = n_nodes
		self.n_in = n_in

		self.weights = self.init_weights(n_in, n_nodes)
		self.f_act, self.f_err = self.set_activation(f_act)

		self.inputs = [0]*(n_in+b)
		self.outputs = [0]*n_nodes
		self.errors = [0]*n_nodes
		self.d_weights = [0]*len(self.weights)


	def init_weights(self, n_in, n_nodes):

		b = self.b 
		
		return [(random()-0.5) for m in range(n_nodes*(n_in+b))]


	def feed_forward(self, inputs):

		output=[]
		inputs.insert(0,self.b)	
		self.inputs=list(inputs)

		w=0
		for i in range(0,self.n_nodes):
			sum_inputs = 0
			for j in range(0,len(self.inputs)):
				sum_inputs = sum_inputs + (self.inputs[j]*self.weights[w])
				w = w+1
			o = self.f_act(sum_inputs)
			output.append(o)
		
		self.output=list(output)
		return output


	def back_propagate(self, out_errors, lr=0.3, mtm=0.6):

		for n in range(len(self.errors)):
			self.errors[n] = self.f_err(self.outputs[n]) * out_errors[n]

		w=0
		d0_weights=list(self.d_weights)

		for n in range(len(self.errors)):
			for m in range(len(self.inputs)):
				self.d_weights[w] = (lr*self.errors[n]*self.inputs[m]) + (mtm*d0_weights[w])
				self.weights[w] = self.weights[w] + self.d_weights[w]
				w = w+1

		return None


	def set_activation(self, f_act= 'sigmoid'):

		def sigmoid(net):	return 1/(1+e**-net)
		def d_sigmoid(x):	return x*(1-x)

		if f_act == 'sigmoid':
			return sigmoid, d_sigmoid


	def __repr__(self):
		rep = "Layer %s with %d weights\n" % (self.name, len(self.weights))
		rep += "Weights:\n"

		for n in range(self.n_nodes):
			i = n*(self.n_in+self.b)
			j = i+self.n_in+1
			rep += "%s_%-5d %s\n" % (self.name, n+1, map(pf,self.weights[i:j])) 

		return rep

def train(ann, all_train_in, all_train_out,epoch, lr, mtm):

	acc = 0
	max_acc = 400
	counter = 0

	while( acc <= len(all_train_in) and epoch>0 ):
		epoch-=1
		if(acc > max_acc):
			max_acc = acc
			save(ann,"final_max.net")
		for data in zip(all_train_in,all_train_out):
			final_output = ann.feed_forward(data[0])
			ann.back_propagate(data[1], lr, mtm)

		save(ann, "final1.net")
		acc = test("final1.net",all_train_in,all_train_out)

	return ann

def test(ann_name,inputs,outputs):
	ann = load(ann_name)
	acc=0
	target=len(inputs)
	for n in range(len(inputs)):
		classified = ann.classify(inputs[n], outputs[n])	
		
		if classified == outputs[n]:
			acc = acc + 1


	print "%d/"%acc, target

	return acc

def scan(ann_name,inputs,outputs):
	ann = load(ann_name)
	result = ''

	classified = ann.classify(inputs,outputs)

	for i in range(len(classified)):
		if(classified[i] == 1):
			result+="NUMBER IS POSSIBLY: " + str(i)

	return result

def save(ann, filename):
	cloud.serialization.cloudpickle.dump(ann, open(filename, "wb"))

def load(filename):
	return pickle.load(open(filename, "rb" ))

def ImageOpen(path):
    
    array = []
    
    try:
        image_file = Image.open(path)
        image_file = image_file.resize((28,28),Image.ANTIALIAS)
        image_file = image_file.convert('L')
        image_file.save("result.png")
        array = np.array(image_file)
    except Exception:
        print "IMAGE NOT FOUND!"
    
    return array

def convertTo1dBinary(array):
    
    array1D = []
    
    for i in range(len(array)):
        for j in range(len(array)):
            if(array[i][j] == 255):
                array1D.append(0)
            else:
                array1D.append(1)
    
    return array1D

def input(path):
    
    input_image = ImageOpen(path)
    converted_input = convertTo1dBinary(input_image)
    
    return converted_input

def createString(i):

	output = ''

	for x in range(5-len(str(i))):
		output+='0'
	output+=str(i)

	return output

def createImg(i):


	output = 'img'

	for x in range(3-len(str(i))):
		output+='0'
	output+=str(i)

	return output

    
def main():

    net = Network([784,15,10])    
    training_inputs, training_outputs = [],[]
    
    inputs = []
    for i in range(10):
        for j in range(100):
            inputs.append(createImg(i+1)+"-"+createString(j+1)+".png")

    outputs = []
    outputs.append([1,0,0,0,0,0,0,0,0,0])
    outputs.append([0,1,0,0,0,0,0,0,0,0])
    outputs.append([0,0,1,0,0,0,0,0,0,0])
    outputs.append([0,0,0,1,0,0,0,0,0,0])
    outputs.append([0,0,0,0,1,0,0,0,0,0])
    outputs.append([0,0,0,0,0,1,0,0,0,0])
    outputs.append([0,0,0,0,0,0,1,0,0,0])
    outputs.append([0,0,0,0,0,0,0,1,0,0])
    outputs.append([0,0,0,0,0,0,0,0,1,0])
    outputs.append([0,0,0,0,0,0,0,0,0,1])
    
    for i in range(10):
        for j in range(100):
            training_outputs.append(outputs[i])

    for j in range(1000):
        training_inputs.append(input("C:/Users/fcsaycon/Desktop/img/"+str(inputs[j])))

    ann = train(net,training_inputs, training_outputs, epoch=100000, lr=0.1, mtm=0.0)
    save(ann, "final1.net")
    test("final1.net",training_inputs,training_outputs)

if __name__ == '__main__':
	main()

    