
import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

# open method used to open different extension image file

#%%

train_dir = os.path.join("data", "train")
#%%
train_dataset = datasets.ImageFolder(train_dir)
#%%
flipped = torchvision.transforms.functional.hflip(train_dataset[0][0])
plt.figure(figsize=(15, 15))
plt.imshow(flipped)
plt.show()
#%%
plt.figure(figsize=(15, 15))
plt.imshow(train_dataset[800][0])
plt.show()
#%%
Atar_try = Image.open('/home/student/HW2/data/500_each_EMNIST-based-Roman/test/iii/3_cap_462.png')


#%%
from skimage.transform import rotate, AffineTransform, warp
# %%
from PIL import Image
# %%
import matplotlib.pyplot as plt
#%%
import numpy as np
#%%
image = Image.open('C:/Users/atarc/Downloads/images for lab/Final_images/ix/img39.png')
#%%
plt.figure()
plt.imshow(image)
plt.show()
#%%
# apply rotate operation
max_rotate = 10
ang = np.random.uniform(low=-max_rotate, high=max_rotate, size=(1)).astype(float)[0]
rotated = torchvision.transforms.functional.rotate(image, ang)
#%%
# apply shift operation
max_rotate = 10
ang = np.random.uniform(low=-max_rotate, high=max_rotate, size=(1)).astype(float)[0]
max_shift = 3.5
up = np.random.uniform(low=-max_shift, high=max_shift, size=(1)).astype(float)[0]
left = np.random.uniform(low=-max_shift, high=max_shift, size=(1)).astype(float)[0]
shifted = torchvision.transforms.functional.affine(image,angle=ang,translate=[up,left],scale=1,shear=[0],fillcolor=[0,0,0])
#%%
plt.figure()
plt.imshow(shifted)
plt.show()
#%%
plt.figure()
plt.imshow(rotated)
plt.show()
#%%
import skimage.io as io
#%%
image = io.imread('C:/Users/atarc/Downloads/images for lab/Final_images/ix/img39.png')
#%%
# shape of the image
print(image.shape)
#%%
# displaying the image
io.imshow(image)
#%%
print('Rotated Image')
#rotating the image by 45 degrees
rotated = rotate(image, angle=45, mode = 'wrap')
#%%
io.imshow(rotated)
#%%
imshow(rotated)
#%%
help(rotate)

#%%
plt.imshow(Atar_try)
plt.show()
#%%
data_transforms = transforms.Compose([transforms.Resize([64, 64])])
resized_atar_try = data_transforms(Atar_try)
plt.figure()
plt.imshow(resized_atar_try)
plt.show()