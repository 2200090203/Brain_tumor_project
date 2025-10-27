from PIL import Image
import numpy as np
from app import is_probable_mri

# MRI sample
mri = Image.open('dataset_images/yes/y0.jpg')
score_mri = is_probable_mri(mri)

# random color image
rand = Image.fromarray((np.random.rand(224,224,3)*255).astype('uint8'))
score_rand = is_probable_mri(rand)

print('MRI score:', score_mri)
print('Random image score:', score_rand)
