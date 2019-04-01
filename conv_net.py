import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ConvNet(object):
  """
  A multi-layer fully-connected neural network. The net has a series of hidden layers, 
  and performs classification over C classes. We train the network with a softmax loss function 
  and L2 regularization on the weight matrices. The network uses ReLU nonlinearity after 
  the fully connected layers.

  """

  def __init__(self, input_size, output_size, filter_size, pooling_schedule, fc_hidden_size,  weight_scale=None, centering_data=False, use_dropout=False, use_bn=False):

    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to 0.01. 
    Inputs:
    - input_size: the dimension D of the input data.
    - hidden_size: a list of sizes of hidden node layers. Each element is the number of hidden nodes in that node layer
    - output_size: the number of classes C.
    - weight_scale: the scale of weight initialization
    - centering_data: whether centering the data or not
    - use_dropout: whether use dropout layers. Dropout rates will be specified in training
    - use_bn: whether to use batch normalization

    Return: 
    """
    tf.reset_default_graph()

    # record all options
    self.options = {'centering_data':centering_data, 'use_dropout':use_dropout, 'use_bn':use_bn}

    #initialize W and b parameters here
    self.params = {'W': [], 'b': []}

    #intial image sizes
    h = input_size[0]
    w = input_size[1]
    c = input_size[2]

    #set pooling schedule as a class member
    self.poolingsched = set(pooling_schedule)

    #for every convolutional layer. intitialize params
    for ind, i in enumerate(filter_size):
      hh = i[0] 
      ww = i[1]
      f = i[2]
      
      wshape = [hh,ww,c,f]
      bshape = f

      if weight_scale is None:
          weight_scale = np.sqrt(2 / (hh*ww*c*f) ) #initialize the weights to something relative to their size. sorta makes sense

      W = tf.Variable(weight_scale * np.random.randn(hh,ww,c,f), dtype=tf.float32)
      b = tf.Variable(0.01 * np.ones(bshape), dtype=tf.float32)
      self.params['W'].append(W)
      self.params['b'].append(b)



      h, w, c = self.calcnewsize(h, w, hh, ww, f)
      
      
      if (ind in self.poolingsched):
        h= int(h/2)
        w = int(w/2)
        #calculate the paramas after pooling

    
    convoutdims = int(h*w*c)
    fc_hidden_size = [convoutdims] + fc_hidden_size + [output_size]

    #for every fc layer initialize parameters
    for i in range( len(fc_hidden_size) - 1):      
      if weight_scale is None:
        weight_scale = np.sqrt(2 / fc_hidden_size[i])

      W = tf.Variable(weight_scale * np.random.randn(fc_hidden_size[i], fc_hidden_size[i + 1]), dtype=tf.float32)
      b = tf.Variable(0.01 * np.ones(fc_hidden_size[i + 1]), dtype=tf.float32)

      self.params['W'].append(W)
      self.params['b'].append(b)
      

    self.convlayers = len(filter_size)
    self.fclayers = len(fc_hidden_size) - 1

    # allocate place holders 
    
    self.placeholders = {}

    # data feeder
    self.placeholders['x_batch'] = tf.placeholder(dtype=tf.float32, shape=[None, input_size[0],input_size[1],input_size[2] ])
    self.placeholders['y_batch']= tf.placeholder(dtype=tf.int32, shape=[None])

    # the working mode 
    self.placeholders['training_mode'] = tf.placeholder(dtype=tf.bool, shape=())
    
    # data center 
    #NOTE changed
    self.placeholders['x_center'] = tf.placeholder(dtype=tf.float32, shape=[input_size[0],input_size[1],input_size[2]])

    # keeping probability of the droput layer
    self.placeholders['keep_prob'] = tf.placeholder(dtype=tf.float32, shape=[])

    # regularization weight 
    self.placeholders['reg_weight'] = tf.placeholder(dtype=tf.float32, shape=[])


    # learning rate
    self.placeholders['learning_rate'] = tf.placeholder(dtype=tf.float32, shape=[])
    
    self.operations = {}

    # construct graph for score calculation 
    scores = self.compute_scores(self.placeholders['x_batch'])
                            
    # predict operation
    self.operations['y_pred'] = tf.argmax(scores, axis=-1)


    # construct graph for training 
    objective = self.compute_objective(scores, self.placeholders['y_batch'])
    self.operations['objective'] = objective

    minimizer = tf.train.GradientDescentOptimizer(learning_rate=self.placeholders['learning_rate'])
    training_step = minimizer.minimize(objective)

    self.operations['training_step'] = training_step 

    if self.options['use_bn']:
        bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else: 
        bn_update = []
    self.operations['bn_update'] = bn_update

    self.operations['training_step'] = tf.group([training_step, bn_update]) #weird that i had to add this.. hm hm hm

    # maintain a session for the entire model
    self.session = tf.Session()

    self.x_center = None # will get data center at training

    return 

  def calcnewsize(self, oldh, oldw, hh, ww, f, pad = 1, stride = 1):
    new_h =  1 + (oldh + 2*pad -hh)/stride 
    new_w =  1 + (oldw + 2*pad -ww)/stride 

    return oldh, oldw, f #return new dimensions after convolution. currently keeps same h and w, c becomes f


  def softmax_loss(self, scores, y):
    """
    Compute the softmax loss. Implement this function in tensorflow

    Inputs:
    - scores: Input data of shape (N, C), tf tensor. Each scores[i] is a vector 
              containing the scores of instance i for C classes .
    - y: Vector of training labels, tf tensor. y[i] is the label for X[i], and each y[i] is
         an integer in the range 0 <= y[i] < C. This parameter is optional; if it
         is not passed then we only return scores, and if it is passed then we
         instead return the loss and gradients.
    - reg: Regularization strength, scalar.

    Returns:
    - loss: softmax loss for this batch of training samples.
    """
    
    # 
    # Compute the loss

    softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores))

    return softmax_loss


  def regularizer(self):
    """ 
    Calculate the regularization term
    Input: 
    Return: 
        the regularization term
    """
    reg = np.float32(0.0)
    for W in self.params['W']:
        reg = reg + self.placeholders['reg_weight'] * tf.reduce_sum(tf.square(W))
    
    return reg


  def compute_scores(self, X):
    """

    Compute the loss and gradients for a two layer fully connected neural
    network. Implement this function in tensorflow

    Inputs:
    - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.

    Returns:
    - scores: a tensor of shape (N, C) where scores[i, c] is the score for 
              class c on input X[i].

    """
    

    # Unpack variables from the params dictionary
    if self.options['centering_data']:  
        X = X - self.placeholders['x_center']

    hidden = X

    ilayer = 0

    #for every convolutional layer
    for i in range(self.convlayers):
      #get params for particular layer
      W = self.params['W'][ilayer]
      b = self.params['b'][ilayer]
      
      #apply conv layer
      hidden = tf.nn.conv2d(hidden, W, [1,1,1,1], padding = 'SAME') + b 
      #stride of all 1s and padding same

      if self.options['use_bn']:
        #apply batch normalization specific to training mode
        hidden = tf.layers.batch_normalization(hidden, training = self.placeholders['training_mode'] )

      if self.options['use_dropout']:
        #apply dropout
        hidden = tf.nn.dropout(hidden,  self.placeholders['keep_prob'])

      hidden = tf.nn.relu(hidden) #apply relu after conv layer and batch norm/dropout

      #possibly pool
      if ilayer in self.poolingsched:
        #pool with height/width = 2 and stride = 2
        hidden = tf.layers.max_pooling2d(hidden, 2, 2, 'SAME')

      ilayer+=1 #increment layer number
    
    #get the first size of the first fc layer to apply proper reshaping of 4d tensor
    firstfclayersize = self.params['W'][ilayer].shape[0] 
    hidden = tf.reshape(hidden, [-1, firstfclayersize] ) #reshape the 4d tensor to 2d for matmul

    #for every fully connected layer
    for i in range(self.fclayers):
      #get params for particular layer
      W = self.params['W'][ilayer]
      b = self.params['b'][ilayer]


      #perform the linear transformation
      linear_trans = tf.matmul(hidden, W) + b

      #if last layer, done
      if (hidden == self.fclayers - 1):
        hidden = linear_trans

      else:
        #apply batch normalization
        if self.options['use_bn']:
          linear_trans = tf.layers.batch_normalization(linear_trans, training = self.placeholders['training_mode'] )

        #apply dropout
        if self.options['use_dropout']:
          linear_trans = tf.nn.dropout(linear_trans,  self.placeholders['keep_prob'])

        #non linear transformation
        hidden = tf.nn.relu(linear_trans)

      #increment the layer
      ilayer+=1

    #scores is finalized
    scores = hidden



    return scores


  def compute_objective(self, scores, y):
    """
    Compute the training objective of the neural network.

    Inputs:
    - scores: A numpy array of shape (N, C). C scores for each instance. C is the number of classes 
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - reg: a np.float32 scalar

    Returns: 
    - objective: a tensorflow scalar. the training objective, which is the sum of 
                 losses and the regularization term
    """

    # get output size, which is the number of classes
    num_classes = self.params['b'][-1].get_shape()[0]

    y1hot = tf.one_hot(y, depth=num_classes)
    loss = self.softmax_loss(scores, y1hot)

    reg_term = self.regularizer()

    objective = loss + reg_term

    return objective

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=1.0, keep_prob=1.0, 
            reg=np.float32(5e-6), num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - keep_prob: the probability of keeping values when using dropout
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    self.x_center = np.mean(X, axis=0)
    self.batch_size = batch_size


    session = self.session
    session.run(tf.global_variables_initializer())

    # Use SGD to optimize the parameters in self.model
    objective_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):

      b0 = (it * batch_size) % num_train 
      batch = range(b0, min(b0 + batch_size, num_train))

      X_batch = X[batch]
      y_batch = y[batch] 

      feed_dict = {self.placeholders['x_batch']: X_batch, 
                   self.placeholders['y_batch']: y_batch, 
                   self.placeholders['learning_rate']:learning_rate, 
                   self.placeholders['training_mode']:True, 
                   self.placeholders['reg_weight']:reg}

      # Decay learning rate
      learning_rate *= learning_rate_decay


      if self.options['centering_data']:
        feed_dict[self.placeholders['x_center']] = self.x_center

      if self.options['use_dropout']:
        feed_dict[self.placeholders['keep_prob']] = np.float32(keep_prob)

     
      np_objective, _ = session.run([self.operations['objective'], 
                                     self.operations['training_step']], feed_dict=feed_dict)

      objective_history.append(np_objective) 

      if verbose and it % 100 == 0:
        print('iteration %d / %d: objective %f' % (it, num_iters, np_objective))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = np.float32(self.predict(X_batch) == y_batch).mean()
        val_acc = np.float32(self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)


    return {
      'objective_history': objective_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }


  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    np_y_pred = self.session.run(self.operations['y_pred'], feed_dict={self.placeholders['x_batch']: X, 
                                                                       self.placeholders['training_mode']:False, 
                                                                       self.placeholders['x_center']:self.x_center, 
                                                                       self.placeholders['keep_prob']:1.0} 
                                                                       )

    return np_y_pred
  
  def get_params(self):
    learned_params = dict()

    learned_params['filter'] = self.session.run(self.params['W'][0])

    return learned_params

