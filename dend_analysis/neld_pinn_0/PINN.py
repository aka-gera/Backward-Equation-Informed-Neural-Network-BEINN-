

import tensorflow as tf
from tensorflow import keras  
 


DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Define the PINN model
class PINN(keras.Model):
    def __init__(self,Dtype=DTYPE,hidden_layers = 4, neurons_per_layer = 50,
                 n_output=1):
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        super(PINN, self).__init__()
        self.ld_init = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", dtype =Dtype )
        self.ld = tf.keras.layers.Dense(neurons_per_layer, activation="tanh",dtype = Dtype)
        self.ld_last = tf.keras.layers.Dense(n_output, dtype = Dtype)

    def call(self, x):
        x = self.ld_init(x)
        for i in range(int(self.hidden_layers)):
          x = self.ld(x)
        x = self.ld_last(x)

        return x

# Train the model
class aka_train():
    def __init__(self) :
      pass

    def get_grad_back_prop(self,fun,model,fun_coef): 

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model.trainable_variables) 
            loss = fun.call(model,fun_coef) 
        grad = tape.gradient(loss, model.trainable_variables)
        del tape

        return loss, grad
    
    @tf.function
    def train_PINN(self,optimizer,fun,model,fun_coef):
        loss, grads = self.get_grad_back_prop(fun,model,fun_coef)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss
    

