from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    max_mat = np.max(X@W, axis=1, keepdims=True)
    loss_mat_0_top = np.exp((X@W)[np.arange(X.shape[0]),y][:,None] - max_mat)
    loss_mat_0_bot = np.sum(np.exp(X@W - max_mat),axis=1, keepdims=True)
    loss_mat_0 = loss_mat_0_top / loss_mat_0_bot
    loss_mat_1 = -np.log(loss_mat_0) / X.shape[0]
    loss += np.sum(loss_mat_1) 
    loss += reg*np.sum(W**2)

    loss_mat_0_top_full = np.exp((X@W) - max_mat)
    mat_0 = - loss_mat_0_top_full / loss_mat_0_bot
    mat_0[np.arange(X.shape[0]),y] += 1
    full_deriv = (-1/X.shape[0])*X[:,:,None] @ mat_0[:,None,:]

    dW += np.sum(full_deriv,axis=0)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    max_mat = np.max(X@W, axis=1, keepdims=True)
    loss_mat_0_top = np.exp((X@W)[np.arange(X.shape[0]),y][:,None] - max_mat)
    loss_mat_0_bot = np.sum(np.exp(X@W - max_mat),axis=1, keepdims=True)
    loss_mat_0 = loss_mat_0_top / loss_mat_0_bot
    loss_mat_1 = -np.log(loss_mat_0) / X.shape[0]
    loss += np.sum(loss_mat_1) 
    loss += reg*np.sum(W**2)

    loss_mat_0_top_full = np.exp((X@W) - max_mat)
    mat_0 = - loss_mat_0_top_full / loss_mat_0_bot
    mat_0[np.arange(X.shape[0]),y] += 1
    full_deriv = (-1/X.shape[0])*X[:,:,None] @ mat_0[:,None,:]

    dW += np.sum(full_deriv,axis=0)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
