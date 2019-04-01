import numpy as np
import warnings



def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.

    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    """
    mode = bn_param['mode']
    eps = bn_param['eps']
    momentum = bn_param['momentum']
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var'] 

    if mode == 'train':
        batchmean = np.mean(x, axis = 0)
        batchvar = np.var(x, axis = 0) #supposedly potentially inaccurate with single precision, variance is a vector in my case
        batchvareps = batchvar + eps
        batchstd = np.sqrt(batchvareps)
        normalized = (x - batchmean)/batchstd  #assuming proper broadcasting this should work
        
        out = (gamma * normalized) - beta

        running_mean = momentum * running_mean + (1-momentum) * batchmean
        running_var = momentum * running_var + (1-momentum) * batchvar

         
    elif mode == 'test':
        runningstd = np.sqrt(running_var + eps)
        normalized = (x - running_mean)/runningstd
        out = (gamma * normalized) + beta
        

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out




def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])



    if mode == 'train':

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask #element wise multiplication of the mask
        return out

    elif mode == 'test':
        return x


    return out



def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (HH, WW, C, F)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    stride = conv_param['stride']
    pad = conv_param['pad']

    paddedx = np.pad(x, ( (0,) ,(pad,) ,(pad,) ,(0,) )   , 'constant', constant_values = 0) #pad only along h and w
    

    n = x.shape[0]
    oldh = x.shape[1]
    oldw = x.shape[2]
    c = x.shape[3]

    hh = w.shape[0]
    ww = w.shape[1]
    numfilters = w.shape[3]
    new_h = int (1 + (oldh - hh + 2*pad) / stride )
    new_w = int( 1 + (oldw - ww + 2*pad) / stride )

    convolutedx = np.zeros((n, new_h, new_h, numfilters)) #initialize output
    
    #for every data point
    for ind, singlex in enumerate(paddedx):
      #find the new convoluted version of it and append to running list
      convoluted = np.zeros( (new_h,new_w,numfilters) )
      for i in range(0,new_h):
        for j in range(0, new_w):
          for k in range(0, numfilters):
            convoluted[i][j][k] = calcpoint(i,j,k,singlex,stride,w,b)
      
      convolutedx[ind] = convoluted

    return convolutedx


def calcpoint(i,j,k, x, stride, w, b):
  actualw = w[:,:,:,k] #the kth set of ws
  actualb = b[k]
  hh = w.shape[0]
  ww = w.shape[1]

  hstart = stride * i
  wstart = stride * j

  hend = hstart + hh  
  wend = wstart + ww  
  sliced = x[hstart:hend,wstart:wend,:]

  elwisemult = sliced * actualw
  summed = np.sum(elwisemult)
  plusbias = summed + actualb

  return plusbias





def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    """

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_height']
    stride = pool_param['stride']
    n = x.shape[0]
    oldh = x.shape[1]
    oldw = x.shape[2]
    c = x.shape[3]

    new_h = int( 1 + (oldh - pool_height)/stride )
    new_w = int( 1 + (oldw - pool_width)/stride )

    newx = np.zeros((n, new_h, new_w, c)) #intialize output

    #for every datapoint
    for ind, singlex in enumerate(x):
      #initialize pooled datapoint
      pooledsingle = np.zeros( (new_h,new_w,c) )

      #fill every index in pooled datapoint
      for i in range(0, new_h):
        for j in range(0, new_w):
          for k in range(0, c):
            pooledsingle[i][j][k] = calcpoolpoint(i,j,k, singlex, stride, pool_height, pool_width)
      
      newx[ind] = pooledsingle

    return newx



def calcpoolpoint(i,j,k,x,stride,pool_height, pool_width):
  hstart = i * pool_height
  wstart = j * pool_width

  hend = hstart + pool_height
  wend = wstart + pool_width
  sliced = x[hstart:hend,wstart:wend, k] 
  return np.max(sliced)







