# Online Fine-tuning of Diffusion Models


### Flower102

The weights for the pre-trained flowers model can be downloaded [here](https://drive.google.com/file/d/1jawOxXaToKEzoQJ3DA8uMdqNXmZIUC-Z/view?usp=sharing). The weights should be copied to `model_weights/model.pt`. We use the Flowers102 dataset, which can automatically be downloaded using `torchvision`. The original train/val/test split of Flowers102 results in 7169 training images, 1020 validation images and 6149 test images. We train the unconditional diffusion model on the joint train and test dataset and use the validation dataset of testing. This gives us a training dataset consisting of 13318 images.

### Forward Operator 

