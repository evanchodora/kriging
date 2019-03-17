## Python Tool for Training Kriging Surrogate Models

Evan Chodora - echodor@clemson.edu

`python kriging.py -h` for help and options

Can be used with one currently coded Spatial Correlation Function (SCF) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

Makes use of the Spatial Distance calculation functions from SciPy to compute the radial distance matrices for the
radial basis function calculations.
(https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

Makes use of the SciPy optimize toolbox for the MLE minimization process. Specifically, the 'L-BFGS-B' (a limited-memory
(L) Broyden-Fletcher-Goldsharb-Shanno (BFGS) algorithm with bounding constraints (B)) is used.
(https://docs.scipy.org/doc/scipy/reference/optimize.html)
(https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

### Prerequisites
 - Python (version 3 and above)
 - numpy
 - scipy

### Training Example:

`python kriging.py -t train -s standard -x x_data.dat -y y_data.dat -m model.db`

In this example, the Kriging surrogate program is called to train a new surrogate using the input parameter file
`x_data.dat` and the output (response) file `y_data.dat` (see the file format specified below). This trained emulator is
generated using a standard SCF and the model is stored in the file `model.db`. When the program runs,
the RBF weights will be computed and the model paramter objects will be saved as a Python shelve
(https://docs.python.org/3/library/shelve.html) database for later use.

### Prediction Example
The code below creates an output file `y_pred.dat` based on the supplied previously trained surrogate model, `model.db`,
at the query locations specified in `x_pred.dat`:

`python kriging.py -t predict -x x_pred.dat -m model.db`

### Input and Output File Format:
Files can be supplied in a comma-separated value format for `x` (input parameters) and `y` (output/response parameters)
with a header line. Prediction files will be generated in the same format with the number of columns corresponding to
the number of trained response dimensions and the number of rows equal to the number of query locations requested.

Example (`x_data.dat`):

```
x0,x1,x2
1.34,5,5.545
3.21,0.56,9.34
```
