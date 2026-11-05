# Shortcut learning: when a model uses the wrong clues
Source: https://doi.org/10.1371/journal.pmed.1002683

Deep learning models can learn shortcuts: features that happen to match the labels in the training data but have nothing to do with the disease. Zech and colleagues (2018) found that pneumonia models could detect which hospital an X-ray came from, and their performance dropped when tested on X-rays from other hospitals.

In this project, some Grad-CAM heatmaps focused on text markers burned into the image, such as the letter "R" or "KV58", instead of the lungs. This suggests the model may partly rely on such markers. A version trained with center cropping and random erasing to reduce this did not perform better, so this remains a known limitation.
