import numpy as np
from random import shuffle
import builtins


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = np.zeros(W.shape) # initialize the gradient as zero # shape(D,C)

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # X[i] => array of D data points (1,D) , W (D,C) => dot product: (1,C) an array of C data points
        correct_class_score = scores[y[i]] # correct class score is given by choosing a value in the above array which is the correct class score
        for j in range(num_classes):
            if j == y[i]:   #go in the loop, with the exception of when the jth iterations is same as the class label
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i] #increment all columns (except the correct class column) by the image pixel array
                dW[:, y[i]] -= X[i] #decrement correct class column by the image pixel array


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train #divide by num train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * W * reg #2WLambda

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

def huber_loss_naive(W, X, y, reg):
    """
    Modified Huber loss function, naive implementation (with loops).
    Delta in the original loss function definition is set as 1.
    Modified Huber loss is almost exactly the same with the "Hinge loss" that you have 
    implemented under the name svm_loss_naive. You can refer to the Wikipedia page:
    https://en.wikipedia.org/wiki/Huber_loss for a mathematical discription.
    Please see "Variant for classification" content.
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    ###############################################################################
    # TODO:                                                                       #
    # Complete the naive implementation of the Huber Loss, calculate the gradient #
    # of the loss function and store it dW. This should be really similar to      #
    # the svm loss naive implementation with subtle differences.                  #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # X[i] => array of D data points (1,D) , W (D,C) => dot product: (1,C) an array of C data points
        correct_class_score = scores[y[i]] # correct class score is given by choosing a value in the above array which is the correct class score
        for j in range(num_classes):
            if j == y[i]:   #go in the loop, with the exception of when the jth iterations is same as the class label
                continue
            margin = scores[j] - correct_class_score
            if margin > 0:
                if margin <= 1:
                    loss += margin**2
                    dW[:, j] += 2*X[i]*margin
                    dW[:, y[i]] -= 2*X[i]*margin
                else:
                    loss += margin
                    dW[:, j] += X[i]
                    dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train #divide by num train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * W * reg #2WLambda

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0] # N
    scores = np.dot(X,W) # N*C
    correct_class_scores = scores[np.arange(num_train), y] # N

    correct_class_scores_reshaped = correct_class_scores[:, np.newaxis] # N,1
    margins = (scores - correct_class_scores_reshaped + 1) #broadcast to all columns of scores to subtract correct class scores, N*C
    margins[np.arange(num_train), y] = 0
    margins = np.maximum(0,margins)
    loss = np.sum(margins)
    loss /= num_train
    loss += 2 * reg * np.sum(W * W)
    print(loss)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    margins[margins > 0] = 1
    margins[margins < 0] = 0
    margins[np.arange(num_train), y] = -1 * np.sum(margins, axis=1)

    dW = np.dot(X.T,margins) #D*N -- N*C ---> D*C
    dW /= num_train
    dW += 2 * W * reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def huber_loss_vectorized(W, X, y, reg):
    """
    Structured Huber loss function, vectorized implementation.

    Inputs and outputs are the same as huber_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured Huber loss, storing the  #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0] # N
    scores = np.dot(X,W) # N*C
    correct_class_scores = scores[np.arange(num_train), y] # N

    correct_class_scores_reshaped = correct_class_scores[:, np.newaxis] # N,1
    margins = (scores - correct_class_scores_reshaped) #broadcast to all columns of scores to subtract correct class scores, N*C
    loss_margins = np.copy(margins)
    loss_margins[np.arange(num_train), y] = 0
    loss_margins = np.maximum(0,loss_margins)

    loss_margins[loss_margins <= 1] = loss_margins[loss_margins <= 1]**2
    loss = np.sum(loss_margins)

    loss /= num_train
    loss += 2 * reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    margins = np.maximum(0,margins)
    margins[margins > 1] = 1
    margins[margins < 1] *= 2


    margins[np.arange(num_train),y] = -1 * np.sum(margins, axis=1)


    dW = np.dot(X.T,margins) #D*N -- N*C ---> D*C
    dW /= num_train
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW