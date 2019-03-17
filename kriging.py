import argparse
import numpy as np
import shelve
from scipy.spatial.distance import squareform, cdist, pdist
from scipy.optimize import minimize


'''
Python Tool for Training Kriging Surrogate Models

https://github.com/evanchodora/kriging

Evan Chodora (2019)
echodor@clemson.edu

Can be used with one currently coded Spatial Correlation Function (SCF) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

Makes use of the Spatial Distance calculation functions from SciPy to compute the radial distance matrices for the
radial basis function calculations.
(https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

Makes use of the SciPy optimize toolbox for the MLE minimization process. Specifically, the 'L-BFGS_B' (a limited-memory
(L) Broyden-Fletcher-Goldsharb-Shanno (BFGS) algorithm with bounding constraints (B)) is used.
(https://docs.scipy.org/doc/scipy/reference/optimize.html)
(https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

Program Usage:

kriging.py [-h] -t {train,predict} [-x X_FILE] [-y Y_FILE] [-m MODEL_FILE] [-s SCF]

optional arguments:
  -h, --help                                    Show help message and exit.
  -t {train,predict}, --type {train,predict}    Specify whether the tool is to be used for training
                                                with "train" or making predictions with a stored model
                                                with "predict". (REQUIRED)
  -x X_FILE                                     Input file of x locations. (OPTIONAL) Default is "x_train.dat".
  -y Y_FILE                                     Output file for surrogate training. (OPTIONAL) Default is
                                                "y_train.dat".
  -m MODEL_FILE, --model MODEL_FILE             File to save the model output or use a previously
                                                trained model file. (OPTIONAL) Default is "model.db".
  -s SCF, --scf SCF                             Specified Spatial Correlation Function to use when
                                                training the surrogate. (OPTIONAL) Default is "standard".

'''


# Class to create or use a Kriging surrogate model
class Kriging:

    # Function to compute beta for the ordinary Kriging algorithm
    def _compute_b(self):
        o = np.ones((self.x_data.shape[0], 1))  # Create a matrix of ones for calculating beta
        beta = np.linalg.multi_dot([o.T, self.r_inv, o, o.T, self.r_inv, self.y_data])
        return beta

    # Function to compute the specified SCF
    def _scf_compute(self, dist):
        if self.scf_func == 'standard':
            r = np.exp(-1 * self.theta * dist ** self.p)
            return r
        else:
            raise ValueError("Not a currently coded SCF")

    # Function to compute the distance matrices - r = (x-c)
    def _compute_dist(self, a, b=None):
        if b is not None:
            # Return the distance matrix between two matrices (or vectors)
            return cdist(a, b, 'minkowski', p=2)
        else:
            # Return a square matrix form of the the pairwise distance for the training locations
            return squareform(pdist(a, 'minkowski', p=2))

    # Function to compute the inverse of the R (SCF) matrix for the Kriging formula
    def _compute_r_inv(self, a, b=None):
        dist = self._compute_dist(a, b)  # Compute the Euclidean distance matrix
        r = self._scf_compute(dist)  # Compute the SCF and return R inverse
        return np.linalg.inv(r)

    # Function to calculate the Maximum Likelihood Estimator for use in the SCF parameter optimization
    def _maximum_likelihood_estimator(self, x):
        self.theta = x[0]  # Assign the passed theta to the current value
        self.p = x[1]  # Assign the passed p to the current value
        self.r_inv = self._compute_r_inv(self.x_data)   # Compute new r_inv
        self.beta = self._compute_b()  # Compute new beta
        n = self.x_data.shape[0]  # Number of training data points
        y_b = self.y_data - np.matmul(np.ones((self.y_data.shape[0], 1)), self.beta)  # Compute (y - ones * beta)
        sigma_sq = (1 / n) * np.matmul(np.matmul(y_b.T, self.r_inv), y_b)  # Compute sigma^2
        # sigma_sq represents the the multivariate covariance matrix between the outputs so should take the determinant
        # of that matrix in order to apply the mle equation (if there is only one output then sigma_sq is a 1x1 matrix
        # so this approach still applies fine)
        mle = n * np.log(np.linalg.det(sigma_sq)) + np.log(np.linalg.det(np.linalg.inv(self.r_inv)))  # Compute MLE
        return mle

    # Function to train a Kriging surrogate using the suplied data and options
    def _train(self):
        x0 = np.array([self.theta, self.p])  # Create array of initial theta and p values for optimization
        # Function to minmize the Maximum Likelihood Estimator to solve for theta and p
        # Chose a limited-memory (L) Broyden-Fletcher-Goldsharb-Shanno (BFGS) algorithm with bounding constraints (B)
        results = minimize(self._maximum_likelihood_estimator, x0, method='L-BFGS-B',
                           bounds=((0.01,10), (0.1,1.99)), options={'gtol': 1e-8})
        self.r_inv = self._compute_r_inv(self.x_data)  # Compute and store R inverse in the class for further us
        self.beta = self._compute_b()  # Compute and store beta in the class for future use

    # Function to use a previously trained Kriging surrogate for predicting at new x locations
    # y_pred = beta + r.T * r_inv * (y - ones * beta)
    def _predict(self):
        r = self._scf_compute(self._compute_dist(self.x_train, self.x_data)).T  # Find r using the prediction locations
        y_b = self.y_data - np.matmul(np.ones((self.y_data.shape[0], 1)), self.beta)  # Compute (y - ones * beta)
        self.y_pred = self.beta + np.linalg.multi_dot([r, self.r_inv, y_b])  # Compute predictions at the locations

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

            # Store model parameters in a Python shelve database
            db = shelve.open(self.model_db)
            db['x_train'] = self.x_data
            db['y_train'] = self.y_data
            db['beta'] = self.beta
            db['r_inv'] = self.r_inv
            db['theta'] = self.theta
            db['p'] = self.p
            db.close()

        else:
            # Read previously stored model data from the database
            model_data = shelve.open(model_db)
            self.x_train = model_data['x_train']
            self.y_data	= model_data['y_train']
            self.beta = model_data['beta']
            self.r_inv = model_data['r_inv']
            self.theta = model_data['theta']
            self.p = model_data['p']
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
    print(surrogate.theta, surrogate.p)
