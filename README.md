# Legendre Decomposition for Tensors
An implementation of Legendre decomposition for tensors, which decomposes a given nonnegative tensor into a multiplicative combination of parameters.
Current implementation supports only third order tensors and zero or nonnegative values are directly ignored.
Please see the following paper for more details:
* Sugiyama, M., Nakahara, H., Tsuda, K.: **Legendre Decomposition for Tensors**, NIPS 2018 (to appear).

## Usage
### In your program
You can perform tensor balancing by calling the function `LegendreDecomposition`.
To use it, you just need to include the header file "legendre_decomposition.h" in your program.
The code is written in C++11 and the [Eigen](http://eigen.tuxfamily.org) library is needed.  

The main function `LegendreDecomposition` is defined as:
```
double LegendreDecomposition(Tensor& X, Int core_size, double error_tol, double rep_max, bool verbose, int type, int const_type, int *num_param)
```
* `X`: an input tensor, the type `Tensor` is defined as `vector<vector<vector<double>>>`  
* `core_size`: the parameter for a decomposition basis  
* `error_tol`: error tolerance  
* `rep_max`: the maximum number of iteration  
* `verbose`: the verbose mode if true  
* `type`: type of an algorithm  
  * `type == 1` Natural gradient (recommended)  
  * `type == 2` Gradient descent  
* `const_type`: type of a decomposition basis (currently can be 1 [complex] or 2 [simple])  
* `num_param`: the number of parameters will be returned  
* Return value: the number of iterations

### In terminal
We provide a sample tensor "test.csv" and a test code "main.cc" to try the code, which includes an input and output interface for tensor files.

For example, in the directory `src/cc`:
```
$ make
$ ./ld -d 4 -i test.csv -c 2
> Read a database file "test.csv":
  Size: (3, 5, 4)
        (Note: this is treated as (4, 3, 5) inside the implementation)
> Start Legendre decomposition by natural gradient:
  Number of parameters: 19
  Step  1, Residual: 0.00419722
  Step  2, Residual: 2.08951e-05
  Step  3, Residual: 5.15477e-10
> Profile:
  Number of iterations: 3
  Running time:         0.000205 [sec]
  RMSE:                 0.222512
```
To compile the program, please edit paths in the "Makefile" according to the location of the Eigen library in your environment.

#### Command-line arguments
* `-i <input_file>`: a path to a csv file of an input tensor (without row and column names)  
* `-o <output_matrix_file>`: an output file of the reconstructed tensor  
* `-t <output_stat_file>`: an output file of statistics  
* `-e <error_tolerance>`: error tolerance is set to 1e-`<error_tolerance>` [default value: 5]  
* `-r <max_iteration>`: the maximum number of iterations is set to 1e+`<max_iteration>` [default value: 6]  
* `-v`: the verbose mode if specified  
* `-n`: the natural gradient is used  
* `-g`: the gradient descent is used  
* `-d`: the depth size (the number of matrices)  
* `-c`: the parameter for a decomposition basis  
* `-b`: type of a decomposition basis [default value: 1]  

## Contact
Author: Mahito Sugiyama  
Affiliation: National Institute of Informatics, Tokyo, Japan  
E-mail: mahito@nii.ac.jp
