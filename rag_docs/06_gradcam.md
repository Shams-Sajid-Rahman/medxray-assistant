# What the Grad-CAM heatmap shows
Source: https://arxiv.org/abs/1610.02391

Grad-CAM (Gradient-weighted Class Activation Mapping) is a method that shows which parts of an image most influenced a neural network's prediction. It uses the gradients flowing into the last convolutional layer to build a coarse heatmap. Red and yellow areas had the strongest influence on the predicted class.

The heatmap is not a map of where disease is. It shows where the model looked, and it can highlight areas that are medically irrelevant. It is also low resolution: for this model it is computed on a 7 by 7 grid and then stretched to the image size, so it shows broad regions, not exact lesions.
