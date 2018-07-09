# Digit Recognizer

# Installation
I was an amateur when I created the setup.py. I'm going to fix it soon.
Run python setup.py

# Usage
Run python DigitRecognizer.py
Choose any of the neurel net configs saved:
28x28-98.4%-500inputs.net
28x28-99.9%-1000inputs.net 

I. Title And Goal
Our project is entitled DigitRecognizer ver. Noob1.1
Our goal is to implement a digit recognizer using ANN given several 128x128 black and white images of single digits. We used at least 10 and a max of 50 images from each number from 0-9. It should be capable of at least recognizing several images of digits of different font faces even if it is Italized or Bolded. 

II. Implementation
A. Data Gathering

We downloaded a data set online at http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
specifically, EnglishFt.tar. It contains 1016 of different images of a single digit. For timeframe constraints, we only chose 10-50 each digit.

B. Preprocessing

We used the Python numpy, PIL module to convert the images to grayscale and then pure black and white images. Using the numpy module, we created a numpy arrray wherein it would create a 2d array of True and False values, True being white and False being black. We converted that array to a 1-D array with True being 0 and False being 1 as inputs for our ANN. 

C. Feature Selection

We used the black and white values of the images as features to train our ANN.

D. Training

We trained our ANN separately thus having several .net files with different attributes. We experimented on which .net file will produce desirable results. We trained our ANN with different constant learning rates. The learning rate is low between [0.1,0.3]. Momentum remained 0. We used the backpropagation algorithm to train our ANN. 
We resized the image to either 64x64 , 32x32, 28,28. We also used only 1-2 hidden layers in our trainings.

E. Results

We got 51%,81%,92%,53% as results in our accuracy from testing all four .net files. But weirdly enough, we needed to feed-forward a few times to get the desired output. Which could be caused by the configuration of the .net file.  But bottomline, it gets the right output for a while.

F. Future Recommendation
Time is needed in this project. Train more samples and use better learning algorithms because back-propagation could take a while. Use a better saving network file. 
We should really use real data not just fonted data.
There's no image processing. No way to focus on a certain frame to remove all that white space. 

Topics
ANN
Image-processing

RELEVANCE
Using this AI, we can build software programs for analyzing softcopy digits. This could lead to automated recognition systems that most robots will use for analyzing certain characters, in this case digits only. Think of all the possibilities! Though, this is limited to black and white digit pictures , it's a small step towards a general character recognition system that will a major breakthrough in due time. :D

