

import tensorflow as tf    


'''
# Train the model
class aka_train_md():
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
    '''



class aka_train_md():

    def __init__(self):
        pass
    @tf.function
    def train_PINN(self, optimizer, fun, model, fun_coef):

        loss, grads = self.get_grad_back_prop(fun, model, fun_coef)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    @tf.function
    def get_grad_back_prop(self, fun, model, fun_coef):

        with tf.GradientTape() as tape:
            loss = fun.call(model, fun_coef)

        grads = tape.gradient(loss, model.trainable_variables)

        return loss, grads











 
class overdamped_lang_loss():
    def __init__(self): 
        pass
  
    def call(self,model,dc):   

        _,u_t,u_q,u_qq = aka_grad().grad_q(model,dc)  
        dyn_rhs =   tf.reshape(tf.reduce_sum( -dc.force*u_q + (dc.gamma/dc.beta)*u_qq,axis=1),(-1,1))
        loss = aka_grad().loss_fn(u_t,dyn_rhs) 
        loss = loss + aka_grad().loss_fn(model(dc.coord_bound),dc.u_bound) 

        return  loss
     