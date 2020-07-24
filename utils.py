"""utils.py 
Utilities for image processing and data munging 
"""

import os
import numpy as np 
from PIL import Image 


def corrupt_image_snp(img, ratio=0.5, amount=0.004):
    """ Add salt and pepper image corruption.  

        `ratio` is the ratio of salt to pepper.  
        `amount` controls the density of corruption added  
    """


    img = np.array(img)
    row, col = img.shape
    out = np.copy(img)
    
    # Generate Salt '1' noise
    num_salt = np.ceil(amount * img.size * ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in img.shape]
    out[coords] = 255
    
    # Generate Pepper '0' noise
    num_pepper = np.ceil(amount* img.size * (1. - ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in img.shape]
    out[coords] = 0
    
    return Image.fromarray(out) 






def corrupt_image_gaus(img, p=0.5, mean=0, var=0.01): 
    """ Add Gaussian noise to each pixel.  

        `p` scaling factor on noise applied  
        `mean` mean of the normal distribution being sampled  
        `var` variance of the normal distribution being sampled 
    """

    img = np.array(img)  
    row, col = img.shape

    sigma = var**0.5
    
    out = p * 255 * np.random.normal(mean, sigma, (row, col))
   
    out = img + out
    
    return Image.fromarray(out.astype('uint8'))






def build_corrupted_dataset(root_dir, dest_dir, noise_fn, noise_fn_params): 
    """ Create corrupted dataset from images in root_dir.  
    
        `noise_fn` specifies which noise function to use from utils  
        `noise_fn_params` specifies the parameters of the noise function  
    """
    
    for filename in os.listdir(root_dir): 
        pth = os.path.join(root_dir, filename)         
        new_pth = os.path.join(dest_dir, filename)
        
        img = Image.open(pth)
        img_c = noise_fn(img, **noise_fn_params)
        
        img_c.save(new_pth)
        
        