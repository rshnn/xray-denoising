"""residual_network.py 
Utilities for the residual autoencoder network   
"""

import tensorflow as tf  
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Add  
from tensorflow.keras.initializers import glorot_uniform  


def ResidualAutoencoder_V2_64(batch_size=32): 
    """ returns a keras model of the architecture described by this paper: 
        https://web.stanford.edu/class/cs331b/2016/projects/zhao.pdf  

        The required input shape is (64, 64, 1)   
    """
    
    input_shape = (64, 64, 1)
    img_input = Input(shape=input_shape)


    ## Downsampling  
    x = img_input
    x = Conv2D(filters=64, kernel_size=(4, 4), padding='same', kernel_initializer=glorot_uniform(seed=0))(x)
    x = Activation('relu')(x) 
    
    shortcut_1 = x 
    x = Conv2D(filters=64, kernel_size=(8, 8), strides=(2, 2), padding='same', 
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = Activation('relu')(x) 
    
    shortcut_2 = x 
    x = Conv2D(filters=128, kernel_size=(8, 8), strides=(2, 2), padding='same', 
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = Activation('relu')(x) 
    
    shortcut_3 = x     
    x = Conv2D(filters=256, kernel_size=(8, 8), strides=(2, 2), padding='same', 
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = Activation('relu')(x) 

    shortcut_4 = x 
    x = Conv2D(filters=518, kernel_size=(4, 4), strides=(2, 2), padding='same', 
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = Activation('relu')(x) 


    
    ## Upsampling 
    x = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding='same')(x) 
    x = Activation('relu')(x) 

    x = Add()([x, shortcut_4]) 
    x = Conv2DTranspose(filters=128, kernel_size=8, strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x) 

    x = Add()([x, shortcut_3]) 
    x = Conv2DTranspose(filters=64, kernel_size=16, strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x) 

    x = Add()([x, shortcut_2]) 
    x = Conv2DTranspose(filters=64, kernel_size=16, strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x) 

    x = Add()([x, shortcut_1])
    x = Conv2DTranspose(filters=1, kernel_size=64, padding='same')(x)    

    return tf.keras.Model(img_input, x, name='res_autoencoder_v2')
    