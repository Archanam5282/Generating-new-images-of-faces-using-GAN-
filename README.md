# Generating-new-images-of-faces-using-GAN

## Description
Face Generation
In this project, you'll define and train a DCGAN on a dataset of faces. Your goal is to get a generator network to generate new images of faces that look as realistic as possible!

The project will be broken down into a series of tasks from loading in data to defining and training adversarial networks. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.

## Flow of the code
### 1. Get the Data

You'll be using the CelebFaces Attributes Dataset (CelebA) to train your adversarial networks. http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

This dataset is more complex than the number datasets (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.

### 2. Pre-processed Data

Since the project's main focus is on building the GANs, we've done some of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.

### 3.Visualize the CelebA Data

The CelebA dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with 3 color channels (RGB)#RGB_Images) each.

Pre-process and Load the Data
Since the project's main focus is on building the GANs, we've done some of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This pre-processed dataset is a smaller subset of the very large CelebA data.

There are a few other steps that you'll need to transform this data and create a DataLoader.

Exercise: Complete the following get_dataloader function, such that it satisfies these requirements:
Your images should be square, Tensor images of size image_size x image_size in the x and y dimension.
Your function should return a DataLoader that shuffles and batches these Tensor images.
ImageFolder
To create a dataset given a directory of images, it's recommended that you use PyTorch's ImageFolder wrapper, with a root directory processed_celeba_small/ and data transformation passed in.

### 4. Create a DataLoader
Exercise: Create a DataLoader celeba_train_loader with appropriate hyperparameters.
Call the above function and create a dataloader to view images.

You can decide on any reasonable batch_size parameter
Your image_size must be 32. Resizing the data to a smaller size will make for faster training, while still creating convincing images of faces!

### 5. Define the Model
A GAN is comprised of two adversarial networks, a discriminator and a generator.

Discriminator
Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with normalization. You are also allowed to create any helper functions that may be useful.

Exercise: Complete the Discriminator class
The inputs to the discriminator are 32x32x3 tensor images
The output should be a single value that will indicate whether a given image is real or fake


### 6. Generator
The generator should upsample an input and generate a new image of the same size as our training data 32x32x3. This should be mostly transpose convolutional layers with normalization applied to the outputs.

Exercise: Complete the Generator class
The inputs to the generator are vectors of some length z_size
The output should be a image of shape 32x32x3


### 7. Initialize the weights of your networks
To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the original DCGAN paper, they say:

All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, your next task will be to define a weight initialization function that does just this!

You can refer back to the lesson on weight initialization or even consult existing model code, such as that from the networks.py file in CycleGAN Github repository to help you complete this function.

Exercise: Complete the weight initialization function
This should initialize only convolutional and linear layers
Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
The bias terms, if they exist, may be left alone or set to 0.


### 8. Build complete network¶
Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.

### 9. Training on GPU
Check if you can train on GPU. Here, we'll set this as a boolean variable train_on_gpu. Later, you'll be responsible for making sure that

Models,
Model inputs, and
Loss function arguments
Are moved to GPU, where appropriate.

### 10. Discriminator and Generator Losses
Now we need to calculate the losses for both types of adversarial networks.

Discriminator Losses
For the discriminator, the total loss is the sum of the losses for real and fake images, d_loss = d_real_loss + d_fake_loss.
Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.
Generator Loss
The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to think its generated images are real.

Exercise: Complete real and fake loss functions
You may choose to use either cross entropy or a least squares error loss to complete the following real_loss and fake_loss functions.

### 11. Optimizers¶
Exercise: Define optimizers for your Discriminator (D) and Generator (G)
Define optimizers for your models with appropriate hyperparameters.

### 12. Training
Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.

You should train the discriminator by alternating on real and fake images
Then the generator, which tries to trick the discriminator and should have an opposing loss function
Saving Samples
You've been given some code to print out some loss statistics and save some generated "fake" samples.

### 13. Training loss¶
Plot the training losses for the generator and discriminator, recorded after each epoch.

### 14. Generator samples from training
View samples of images from the generator, and answer a question about the strengths and weaknesses of your trained models.
