## Python Tool for Training Kriging Surrogate Models

Evan Chodora - echodor@clemson.edu

`python kriging.py -h` for help and options

Can be used with one currently coded Spatial Correlation Function (SCF) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

### Training Example:

`python kriging.py -t train -s standard -x x_data.dat -y y_data.dat -m model.db`

### Prediction Example
Creates an output file `y_pred.dat` based on the supplied input values:

`python kriging.py -t predict -x x_pred.dat -m model.db`

### Input and Output File Format:
Files can be supplied in a comma-separated value format for `x` and `y` with a header line.

Example (`x_data.dat`):

```
x0,x1,x2
1.34,5,5.545
3.21,0.56,9.34
```
