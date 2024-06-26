import builtins
from matplotlib.pyplot import close
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with Cosine and L2 distances """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0, distfn='L2'):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            if distfn == 'L2':
                dists = self.compute_L2_distances_no_loops(X)
            else:
                dists = self.compute_Cosine_distances_no_loops(X)
        elif num_loops == 1:
            if distfn == 'L2':
                dists = self.compute_L2_distances_one_loop(X)
            else:
                dists = self.compute_Cosine_distances_one_loop(X)
        elif num_loops == 2:
            if distfn == 'L2':
                dists = self.compute_L2_distances_two_loops(X)
            else:
                dists = self.compute_Cosine_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_L2_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0] #array with 500 test examples
        num_train = self.X_train.shape[0] #array with 5000 training examples
        dists = np.zeros((num_test, num_train))
        
        #loop 500
        for i in range(num_test):
            # loop 5000
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                ith_test_image = X[i] #3072 array
                jth_train_image = self.X_train[j] #3072 array

                dists[i][j] = np.sqrt(np.sum(np.power(np.subtract(ith_test_image,jth_train_image),2)))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            ith_test_image = X[i] #3072 pixel array

            #self.X_train #(5000,3072) array
            dists[i] = np.sqrt(np.sum(np.power(np.subtract(ith_test_image,self.X_train[:,]),2),axis=1)) #array with 5000 size, each is a different distanc

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #(xi^2 - yi^2) = xi^2 + yi^2 - 2xiyi ==> X_train^2 + X_test^2 - 2 * x_test o x_train.T

        X_test_sq = np.sum(np.power(X,2),axis=1,keepdims=True)

        X_train_sq = np.sum(np.power(self.X_train,2),axis=1,keepdims=True)

        middle_term = (-2 * np.dot(X,self.X_train.T))

        dists = np.sqrt(X_test_sq + X_train_sq.T + middle_term)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_Cosine_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Cosine distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                              #
                # Compute the cosine distance between the ith test point and the jth #
                # training point, and store the result in dists[i, j]. You should    #
                # not use a loop over dimension, nor use np.linalg.norm() and
                # scipy.spatial.distance.cosine
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                

                ith_test_image = X[i] #3072 array
                jth_train_image = self.X_train[j] #3072 array

                dists[i][j] = np.dot(ith_test_image,jth_train_image) / (np.sqrt(np.sum(ith_test_image))*np.sqrt(np.sum(jth_train_image)))
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_Cosine_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_Cosine_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the cosine distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm(). and scipy.spatial.distance.cosine      #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
    
            ith_test_image = X[i] #3072 pixel array

            #one loop cosine
            dists[i] = np.dot(self.X_train,ith_test_image) / (np.sqrt(np.sum(ith_test_image))*np.sqrt(np.sum(self.X_train[:,],axis=1))) #array with 5000 size, each is a different distance

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
    

    def compute_Cosine_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_Cosine_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the cosine distance between all test points and all training  #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm() or scipy.spatial.distance.cosine             #
        #                                                                       #
        #                                                                       #
        #                                                                       #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        

        #non loop cosine

        dot_term = np.dot(X,self.X_train.T)

        eucladian_x1 = np.sqrt(np.sum(X,axis=1,keepdims=True)) #500,1

        eucladian_x2 = np.sqrt(np.sum(self.X_train,axis=1,keepdims=True)).T #1,5000

        normalizer_term = np.dot(eucladian_x1,eucladian_x2)

        dists = np.divide(dot_term,normalizer_term) #500,5000


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #ith test image's distances to 5000 training examples
            ith_test_image_distances = dists[i]
            #find indexes containing the minimum distances (k times)
            min_distances = np.argsort(ith_test_image_distances)[:k]
            #extract their labels via looking at the testing data
            closest_y = self.y_train[min_distances]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


            import collections
            #numbers as keys frequency as values
            freq_table = dict(collections.Counter(closest_y))
            max_occurence = max(freq_table.values())

            y_pred[i] = min([k for k,v in freq_table.items() if float(v) == max_occurence])
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
