
import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

# open method used to open different extension image file

#%%

train_dir = os.path.join("data", "train")
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
plt.figure()
plt.imshow(Atar_try)
plt.show()
#%%
plt.imshow(Atar_try)
plt.show()
#%%
data_transforms = transforms.Compose([transforms.Resize([64, 64])])
resized_atar_try = data_transforms(Atar_try)
plt.figure()
plt.imshow(resized_atar_try)
plt.show()