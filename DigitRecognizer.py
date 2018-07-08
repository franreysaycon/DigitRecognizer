from PIL import Image, ImageTk
import numpy as np
from numpy import *
from random import *
from functools import reduce
from math import *
import operator
import os, sys
import re
import pickle
import copy
from Tkinter import *
from tkFileDialog import askopenfilename
from threading import Thread 
from time import sleep
import PIL
import tkMessageBox
import Tkinter
import tkSimpleDialog

dimensionX = 28
dimensionY = 28

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

	def classify(self, inputs, outputs):

		actual = self.feed_forward(inputs)
		
		classified = [0]*len(actual)
		for i in range(len(classified)):
			if actual[i] > 0.5: classified[i] = 1 
		
		return classified

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

	def set_activation(self, f_act= 'sigmoid'):

		def sigmoid(net):	return 1/(1+e**-net)
		def d_sigmoid(x):	return x*(1-x)

		if f_act == 'sigmoid':
			return sigmoid, d_sigmoid

class GUI(Frame):
  
    def __init__(self, parent,imageOnLoad,ann,result):
        Frame.__init__(self, parent, background="lightblue")   
         
        self.parent = parent
        self.imageOnLoad = imageOnLoad
        self.ann = ann
        self.result = result

        im = PIL.Image.open("background.png")
        tkimage = ImageTk.PhotoImage(im)
        myvar= Label(self,image = tkimage)
        myvar.image = tkimage
        myvar.pack()
        myvar.place(x=0, y=0, relwidth=1, relheight=1)

        self.parent.title("Digit Recognizer ver. Noob1.1")
        self.pack(fill=BOTH, expand=1)

        self.centerWindow()
        self.createButtons()
        self.loadText()

    def centerWindow(self):
      
        w = 400
        h = 600

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        
        x = (sw - w)/2 
        y = (sh - h)/2 - 20
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    def createButtons(self):

    	loadButton = Button(self, height = 2, width=15,text="Load ANN",command=self.openANN)
        loadButton.place(x=20, y=20)
        imageButton = Button(self, height = 2, width = 15, text="Load Image", command = self.openImage)
        imageButton.place(x=20,y=70)
        scanButton = Button(self, height = 2, width = 15, text="Scan!!", command = self.scanImage)
        scanButton.place(x=20,y=120)        
        changeButton = Button(self, height = 2, width = 15, text="Change Dimension", command = self.changeDimension)
        changeButton.place(x=20,y=170)  
        quitButton = Button(self, height = 2, width = 15, text="Quit", command = self.quit)
        quitButton.place(x=20,y=220)   


    def openImage(self):
    	try:
			filename = askopenfilename()
			image = PIL.Image.open(filename)
			photo = ImageTk.PhotoImage(image)
			label = Label(self,image=photo)
			label.image = photo
			label.pack()
			label.place(x=200,y=60)
			self.imageOnLoad = filename
			self.result+='\n'
			self.result+='Successfully opened ' + filename
			self.loadText()
        except Exception:
			tkMessageBox.showinfo("Error","Please load a png file!!!")

    def openANN(self):
		filename = askopenfilename()
		if ".net" in filename:
			self.ann = filename
			self.result+='\n'
			self.result+='Opened .net file from ' + filename
			self.loadText()
		else:
			tkMessageBox.showinfo("Error","Please load a .net file!!!")


    def scanImage(self):

        if not str(dimensionX)+"x"+str(dimensionY) in self.ann and not self.ann is '' :
        	tkMessageBox.showinfo("Error","Wrong Dimensions!!!"+"\n"+"Current Dimension " + str(dimensionX) + "x" + str(dimensionY) + " does not fit!" )
        	return

    	if not self.ann is '' and not self.imageOnLoad is '':
    		results = scan(self.ann,input(self.imageOnLoad),[])
    		self.result +='\n'
	    	self.result +='Scanning now!'
	    	self.result += '\n'
	    	self.result += results
	    	self.loadText()

    	else:
    		tkMessageBox.showinfo("Error","Please load an ann and/or an image file!!!")

    def loadText(self):
    	w = Text(self,font="Garamond", height = 17, width = 46)
    	w.insert(INSERT,self.result)
    	w.place(x=20,y=280)
    	w.config(state=DISABLED)
        scrl = Scrollbar(self, command=w.yview,width = 18)
        w.config(yscrollcommand=scrl.set)
        scrl.pack(fill=BOTH)
        scrl.place(x=0,y=280)
        w.see(Tkinter.END)

    def changeDimension(self):
        global dimensionX
        dimensionX = tkSimpleDialog.askinteger("Change Dimension!", "Enter x value:")
        global dimensionY
        dimensionY = tkSimpleDialog.askinteger("Change Dimension!", "Enter y value:")
        self.result+='\n'
        self.result+="Changed Dimension to " + str(dimensionX) + "x" + str(dimensionY)
        self.loadText()

def scan(ann_name,inputs,outputs):
	ann = load(ann_name)
	result = 'Number is possibly:'

	classified = ann.classify(inputs,outputs)

	for i in range(len(classified)):
		if(classified[i] == 1):
			result+=" " + str(i)

	return result

def load(filename):
	return pickle.load(open(filename, "rb" ))

def ImageOpen(path):
    
    array = []
    
    try:
        image_file = PIL.Image.open(path)
        image_file = image_file.resize((dimensionX,dimensionY),PIL.Image.ANTIALIAS)
        image_file = image_file.convert('L')
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

    root = Tk()
    ex = GUI(root,'','','Hello!!!! There :D')
    root.mainloop()

if __name__ == '__main__':
	main()

    