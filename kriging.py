import argparse
import numpy as np
import shelve
from scipy.spatial.distance import squareform, cdist, pdist


'''
Python Tool for Training Kriging Surrogate Models

https://github.com/evanchodora/kriging

Evan Chodora (2019)
echodor@clemson.edu
'''


# Class to create or use a Kriging surrogate model
class Kriging:

    # Function to compute beta for the ordinary Kriging algorithm
    def _compute_b(self):
        o = np.ones((self.x_data.shape[0], 1))  # Create a matrix of ones for calculating beta
        print(self.r_inv)
        beta = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(o.T, self.r_inv), o), o.T), self.r_inv), self.y_data)
        return beta

    # Function to compute the Euclidean distance (r)
    def _compute_r(self, a, b=None):
        if b is not None:
            # Return the euclidean distance matrix between two matrices (or vectors)
            r = np.exp(-self.theta * cdist(a, b, 'euclidean') ** 1)
            return r
        else:
            # Return a square matrix form of the the pairwise euclidean distance for the training locations
            r = np.exp(-self.theta * squareform(pdist(a, 'euclidean')) ** 1)
            return r

    def _train(self):
        r = self._compute_r(self.x_data)  # Compute the R matrix
        self.r_inv = np.linalg.inv(r)  # Compute and store R inverse in the class for further use
        self.beta = self._compute_b()
        print(self.beta)


    # Initialization for the Kriging class
    # Defaults are specified for the options, required to pass in whether you are training or predicting
    def __init__(self, type, x_file='x_train.dat', y_file='y_train.dat', model_db='model.db', scf_func='standard'):

        self.x_data = np.loadtxt(x_file, skiprows=1, delimiter=",")  # Read the input locations file
        self.x_data = self.x_data.reshape(self.x_data.shape[0], -1)  # Reshape into 2D matrix (avoids array issues)
        self.scf_func = scf_func  # Read user specified options (or the defaults)
        self.model_db = model_db  # Read user specified options (or the defaults)

        # Check for training or prediction
        if type == 'train':
            self.y_data = np.loadtxt(y_file, skiprows=1, delimiter=",")  # Read output data file
            self.y_data = self.y_data.reshape(self.y_data.shape[0], -1)  # Reshape into 2D matrix (avoids array issues)

            # Set initial values for theta and p
            self.theta = 0.5  # 0 < theta
            self.p = 1.0  # 0 < p < 2

            self._train()  # Run the model training function

            # # Store model parameters in a Python shelve database
            # db = shelve.open(self.model_db)
            # db['x_train'] = self.x_data
            # db.close()

        else:
            # Read previously stored model data from the database
            model_data = shelve.open(model_db)
            self.x_train = model_data['x_train']
            model_data.close()

            self._predict()  # Run the model prediction functions

            # Quick loop to add a header that matches the input file format
            y_head = []
            for i in range(self.y_pred.shape[1]):
                y_head.append('y' + str(i))

            # Convert header list of strings to a single string with commas and write out the predictions to a file
            header = ','.join(y_head)
            np.savetxt('y_pred.dat', self.y_pred, delimiter=',', fmt="%.6f", header=header, comments='')

# Code to run when called from the command line (usual behavior)
if __name__ == "__main__":

    # Parse the command line input options to "opt" variable when using on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', dest='type', choices=['train', 'predict'], required=True,
                        help="""Specify whether the tool is to be used for training with \"train\" or
                        making predictions with a stored model with \"predict\".""")
    parser.add_argument('-x', dest='x_file', default='x_train.dat',
                        help="""Input file of x locations. Default is \"x_train.dat\".""")
    parser.add_argument('-y', dest='y_file', default='y_train.dat',
                        help="""Output file for surrogate training. Default is \"y_train.dat\".""")
    parser.add_argument('-m', '--model', dest='model_file', default='model.db',
                        help="""File to save the model output or use a previously trained model file.
                        Default is \"model.db\".""")
    parser.add_argument('-s', '--scf', dest='scf', default='standard',
                        help="""Specified Spatial Correlation Function to use when training the surrogate.
                        Default is \"standard\".""")
    opts = parser.parse_args()

    # Create and run the RBF class object
    surrogate = Kriging(opts.type, opts.x_file, opts.y_file, opts.model_file, opts.scf)
