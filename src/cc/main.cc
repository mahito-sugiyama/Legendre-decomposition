#include "legendre_decomposition.h"
#include <unistd.h>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main(int argc, char *argv[]) {
  bool verbose = false;
  bool flag_in = false;
  bool flag_out = false;
  bool flag_stat = false;
  bool do_natural = true;
  bool do_graddescent = false;
  double rep_max = 1e+06;
  double error_tol = 1e-05;
  char *input_file = NULL;
  char *output_file = NULL;
  char *stat_file = NULL;
  Int num_mat = 1;
  Int core_size = 2;
  Int const_type = 1;

  // get arguments
  char opt;
  while ((opt = getopt(argc, argv, "i:o:t:e:r:vngd:c:b:")) != -1) {
    switch (opt) {
    case 'i': input_file = optarg; flag_in = true; break;
    case 'o': output_file = optarg; flag_out = true; break;
    case 't': stat_file = optarg; flag_stat = true; break;
    case 'e': error_tol = pow(10, -1 * atof(optarg)); break;
    case 'r': rep_max = pow(10, atof(optarg)); break;
    case 'v': verbose = true; break;
    case 'n': do_natural = true; do_graddescent = false; break;
    case 'g': do_natural = false; do_graddescent = true; break;
    case 'd': num_mat = atoi(optarg); break;
    case 'c': core_size = atoi(optarg); break;
    case 'b': const_type = atoi(optarg); break;
    }
  }

  if (!flag_in) {
    cerr << "> ERROR: Input file (-i [input_file]) is missing!" << endl;
    exit(1);
  }
  ofstream sfs;
  if (flag_stat) {
    sfs.open(stat_file);
  }

  cout << "> Read a database file \"" << input_file << "\":" << endl << flush;
  Tensor X;
  ifstream ifs(input_file);
  if (!ifs) {
    cerr << endl << "  ERROR: The file \"" << input_file << "\" does not exist!!" << endl;
    exit(1);
  }
  readTensorFromCSV(X, num_mat, ifs);
  ifs.close();

  cout << "  Size: (" << X.front().size() << ", " << X.front().front().size() << ", " << X.size() << ")" << endl << flush;
  cout << "        (Note: this is treated as (" << X.size() << ", " << X.front().size() << ", " << X.front().front().size() << ") inside the implementation)" << endl << flush;

  Tensor X_org;
  X_org = Tensor(X.size(), vector<vector<double>>(X.front().size(), vector<double>(X.front().front().size(), 0)));
  for (Int i = 0; i < X.size(); ++i) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int k = 0; k < X.front().front().size(); ++k) {
	X_org[i][j][k] = X[i][j][k];
      }
    }
  }

  if (flag_stat) {
    sfs << "Number_of_dim1:\t" << X.front().front().size() << endl;
    sfs << "Number_of_dim2:\t" << X.front().size() << endl;
    sfs << "Number_of_dim3:\t" << X.size() << endl;
  }

  Int type = 1;
  if (do_natural) {
    cout << "> Start Legendre decomposition by natural gradient:" << endl << flush;
    type = 1;
  } else if (do_graddescent) {
    cout << "> Start Legendre decomposition by gradient descent:" << endl << flush;
    type = 2;
  }

  Int num_param;
  // run Legendre decomposition
  auto ts = system_clock::now();
  double step = LegendreDecomposition(X, core_size, error_tol, rep_max, verbose, type, const_type, &num_param);
  auto te = system_clock::now();
  auto dur = te - ts;

  double rmse = computeRMSE(X, X_org);

  cout << "> Profile:" << endl;
  cout << "  Number of iterations: " << step << endl;
  cout << "  Running time:         " << duration_cast<microseconds>(dur).count() / 1000000.0 << " [sec]" << endl;
  cout << "  RMSE:                 " << rmse << endl;
  if (flag_out) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
    ofstream ofs(output_file);
    ofs << X;
    ofs.close();
  }
  if (flag_stat) {
    sfs << "Number_of_constraint:\t" << num_param << endl;
    sfs << "Number_of_iterations:\t" << step << endl;
    sfs << "Running_time_(sec):\t" << duration_cast<microseconds>(dur).count() / 1000000.0 << endl;
    sfs << "RMSE:\t" << rmse << endl;
    sfs.close();
  }
}
