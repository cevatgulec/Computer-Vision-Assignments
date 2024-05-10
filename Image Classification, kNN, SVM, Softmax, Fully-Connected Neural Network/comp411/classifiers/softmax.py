import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W) #1*D,D*C --> 1*C
        correct_class_score = scores[y[i]] # 1

        exp_sum = 0
        for j in range(num_classes):
            exp_sum += np.exp(scores[j])
        loss += -np.log(np.exp(correct_class_score) / exp_sum)
        
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
            if (j == y[i]):
                dW[:, y[i]] -= X[i]     

    loss /= num_train
    loss += reg_l2 * np.sum(W**2)

    dW /= num_train
    dW += 2 * reg_l2 * W

    if regtype == 'ElasticNet':
        loss += reg_l1 * np.sum(np.abs(W))
        dW += reg_l1*np.sign(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = np.dot(X,W) # N*C
    correct_class_scores = scores[np.arange(num_train), y] # N
    exp_scores = np.exp(scores)
    exp_sums = np.sum(exp_scores,axis=1) #N
    loss = np.sum(-correct_class_scores + np.log(exp_sums))

    loss /= num_train
    loss += reg_l2 * np.sum(W**2)

    dW = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    dW /= np.sum(dW, axis=1,keepdims=True)
    dW[np.arange(num_train), y] -= 1
    dW = X.T.dot(dW)

    dW /= num_train
    dW += 2 * reg_l2 * W

    if regtype == 'ElasticNet':
        loss += reg_l1 * np.sum(np.abs(W))
        #dW += reg_l1*np.sign(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
