# MMR-Autoencoder Assignment

[Project Setup](./setup.md)

## Description
Using *Pytorch* and the *CIFAR10* dataset, we train an autoencoder network. 


## Steps:
1. Download and Prepare Data 
2. Build simple autoencoder network ( e.g., two dense layers for both encoder
and decoder, with the ability to split encoder and decoder into two models)
4. Try to train the network
3. Build in model saving and loading features
5. Build tests that analyses distributions of latent space
6. Refine hyper-parameters (layer sizes, parameter for convolutions, or activation functions)

### Testing the model:
- *Visualizing Reconstruction Results:* Plot images next to their reconstruction results
- *Distribution Analysis:* Use encoder only to encode random images from the test set. Visualise using pairplot function. 
- *Projecting Results:* Project results of the test set to 2D using UMAP. Visualize as a scatterplot with color representing the classes. You should see some separately colored clusters.

### Using the model:
Select a random image, and encode it. Encode the rest of the images in the dataset too. Compute the distances pairwise between the picked image and the whole dataset, then sort the images in ascending order. Plot the 10 most similar images. Repeat this for 20 images.


## Bonus
- Preprocess images with noise and a transformation to get more data
- Train model first and then add more and more distorted images
- Compare robustness between the model before and after adding distorted images


## PDF Report:
Write a report of each step with the code and visualize the results.