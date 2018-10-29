// #define EIGEN_USE_MKL_ALL
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <functional>
#include <utility>
#include <iomanip>
#include <tuple>
#include <time.h>
#include <chrono>
#include <Eigen/Dense>

#define Int int32_t
#define Tensor vector<vector<vector<double>>>
#define Poset vector<vector<vector<node>>>
#define PosetIndex vector<vector<pair<Int, Int>>>
#define D 3

using namespace std;
using namespace Eigen;
using namespace std::chrono;

double EPSILON = 1e-300;

// node structure
class node {
public:
  bool nonzero;
  Int id, id_org;
  double p, p_tmp;
  double theta, theta_prev, theta_sum, theta_sum_prev;
  double eta, eta_prev;
};
Poset S_global;

// output a tensor
ostream &operator<<(ostream& out, const Tensor& X) {
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (Int i = 0; i < (Int)vec.size() - 1; ++i) {
	out << vec[i] << ", ";
      }
      out << vec.back() << endl;
    }
    out << endl;
  }
  return out;
}

// for "reverse" in range-based loop
template<class Cont> class const_reverse_wrapper {
  const Cont& container;
public:
  const_reverse_wrapper(const Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() const { return container.rbegin(); }
  decltype(container.rend()) end() const { return container.rend(); }
};
template<class Cont> class reverse_wrapper {
  Cont& container;
public:
  reverse_wrapper(Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() { return container.rbegin(); }
  decltype(container.rend()) end() { return container.rend(); }
};
template<class Cont> const_reverse_wrapper<Cont> reverse(const Cont& cont) {
  return const_reverse_wrapper<Cont>(cont);
}
template<class Cont> reverse_wrapper<Cont> reverse(Cont& cont) {
  return reverse_wrapper<Cont>(cont);
}

// for sort
struct greaterP { bool operator()(const tuple<Int, Int, Int>& b1, const tuple<Int, Int, Int>& b2) const { return S_global[get<0>(b1)][get<1>(b1)][get<2>(b1)].p > S_global[get<0>(b2)][get<1>(b2)][get<2>(b2)].p; } };


// read a database file
void readTensorFromCSV(Tensor& X, Int num_mat, ifstream& ifs) {
  vector<vector<double>> data;
  string line;
  while (getline(ifs, line)) {
    stringstream lineStream(line);
    string cell;
    vector<double> tmp;
    while (getline(lineStream, cell, ',')) {
      tmp.push_back(stod(cell));
    }
    data.push_back(tmp);
  }

  if (data.size() % num_mat != 0) {
    cerr << endl << "The size specification of the input tensor (= " << num_mat << ") is invalid!" << endl;
    exit(1);
  }

  X = Tensor(num_mat, vector<vector<double>>(data.size() / num_mat, vector<double>(data.front().size(), 0)));

  for (Int k = 0; k < num_mat; ++k) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int i = 0; i < X.front().front().size(); ++i) {
	X[k][j][i] = data[j + k * X.front().size()][i];
      }
    }
  }
}

// preprocess X by a one step Sinkhorn balancing
double preprocess(Tensor& X) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  double X_sum = 0.0;
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      X_sum += accumulate(vec.begin(), vec.end(), 0.0);
    }
  }
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x /= X_sum;
      }
    }
  }
  return X_sum;
}
// make a node index
void makePosetIndex(Tensor& X, PosetIndex& idx_tp) {
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};

  // traverse a matrix in the topological order
  for (Int ii = 0; ii < D; ++ii) {
    Int i1 = ii;
    Int i2 = ii + 1; if (i2 > 2) i2 -= D;
    Int i3 = ii + 2; if (i3 > 2) i3 -= D;
    for (Int i = 0; i < n[i2] + n[i3] - 1; ++i) {
      Int j = i + 1 - n[i2];
      if (j < 0) j = 0;
      for (; j <= min(i, n[i3] - 1); j++) {
	idx_tp[i1].push_back(make_pair(i - j, j));
      }
    }
  }
}
// make a node matrix from eigen matrix
void makePosetTensor(Tensor& X, Poset& S) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  // initialization
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	S[i][j][k].p = X[i][j][k];
	S[i][j][k].theta = 0; S[i][j][k].theta_sum = 0; S[i][j][k].theta_sum_prev = 0;
	S[i][j][k].eta = 0;
	S[i][j][k].nonzero = S[i][j][k].p > EPSILON ? true : false;
      }
    }
  }
  S[0][0][0].nonzero = true;
}
// make a beta the submanifold
void prepareForBeta(Poset& S) {
  for (Int i = 0; i < S.size(); ++i) {
    for (Int j = 0; j < S.front().size(); ++j) {
      for (Int k = 0; k < S.front().front().size(); ++k) {
	S[i][j][k].p_tmp = S[i][j][k].p;
      }
    }
  }
  S[0][0][0].p_tmp = -1;
}
void makeBetaCore(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, Int core_size, bool random) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  // for sorting
  S_global.resize(n1);
  for (auto&& s : S_global) {
    s.resize(n2);
    for (auto&& u : s) u.resize(n3);
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	S_global[i][j][k].p = S[i][j][k].p;
      }
    }
  }

  vector<tuple<Int, Int, Int>> univ;
  vector<tuple<Int, Int, Int>> s;
  mt19937 g(1);
  for (Int i = 0; i < n1; ++i) {
    univ.clear(); s.clear();
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	if (S[i][j][k].p_tmp > 0) {
	  univ.push_back(make_tuple(i, j, k));
	}	
      }
    }
    if (random) shuffle(univ.begin(), univ.end(), g);
    else sort(univ.begin(), univ.end(), greaterP());
    if (core_size > univ.size()) core_size = univ.size();
    for (Int c = 0; c < core_size; ++c) {
      tuple<Int, Int, Int>& b = univ[c];
      beta.push_back(make_pair(b, S[get<0>(b)][get<1>(b)][get<2>(b)].eta));
      S[get<0>(b)][get<1>(b)][get<2>(b)].p_tmp = -1.0;
    }
  }
}
void makeBetaNorm(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, Int core_size) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  vector<Int> idx1;
  if (core_size < n1) {
    Int diff1 = n1 / core_size;
    for (Int i = 0; i < core_size; ++i) idx1.push_back(i * diff1);
  } else {
    idx1.resize(n1); iota(idx1.begin(), idx1.end(), 0);
  }
  vector<Int> idx2;
  if (core_size < n2) {
    Int diff2 = n2 / core_size;
    for (Int i = 0; i < core_size; ++i) idx2.push_back(i * diff2);
  } else {
    idx2.resize(n2); iota(idx2.begin(), idx2.end(), 0);
  }
  vector<Int> idx3;
  if (core_size < n3) {
    Int diff3 = n3 / core_size;
    for (Int i = 0; i < core_size; ++i) idx3.push_back(i * diff3);
  } else {
    idx3.resize(n3); iota(idx3.begin(), idx3.end(), 0);
  }


  for (Int i = 1; i < n1; ++i) {
    if (S[i][0][0].p_tmp > 0) {
      beta.push_back(make_pair(make_tuple(i, 0, 0), S[i][0][0].eta));
      S[i][0][0].p_tmp = -1;
    }
  }
  for (Int j = 1; j < n2; ++j) {
    beta.push_back(make_pair(make_tuple(0, j, 0), S[0][j][0].eta));
    S[0][j][0].p_tmp = -1;
  }
  for (Int k = 1; k < n3; ++k) {
    beta.push_back(make_pair(make_tuple(0, 0, k), S[0][0][k].eta));
    S[0][0][k].p_tmp = -1;
  }

  for (auto&& i : idx1) {
    for (auto&& j : idx2) {
      if (S[i][j][0].p_tmp > 0) {
	beta.push_back(make_pair(make_tuple(i, j, 0), S[i][j][0].eta));
	S[i][j][0].p_tmp = -1;
      }
    }
    for (auto&& k : idx3) {
      if (S[i][0][k].p_tmp > 0) {
	beta.push_back(make_pair(make_tuple(i, 0, k), S[i][0][k].eta));
	S[i][0][k].p_tmp = -1;
      }
    }
  }
}
// compute eta for all entries
void computeEta(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  vector<Int> idx_row(n1 - 1);
  vector<Int> idx_col(n2 - 1);
  vector<Int> idx_dep(n3 - 1);
  iota(idx_row.begin(), idx_row.end(), 0);
  iota(idx_col.begin(), idx_col.end(), 0);
  iota(idx_dep.begin(), idx_dep.end(), 0);

  S[idx_row.size()][idx_col.size()][idx_dep.size()].eta = S[idx_row.size()][idx_col.size()][idx_dep.size()].p;
  for (auto&& i : reverse(idx_row))
    S[i][idx_col.size()][idx_dep.size()].eta = S[i][idx_col.size()][idx_dep.size()].p + S[i + 1][idx_col.size()][idx_dep.size()].eta;
  for (auto&& j : reverse(idx_col))
    S[idx_row.size()][j][idx_dep.size()].eta = S[idx_row.size()][j][idx_dep.size()].p + S[idx_row.size()][j + 1][idx_dep.size()].eta;
  for (auto&& k : reverse(idx_dep))
    S[idx_row.size()][idx_col.size()][k].eta = S[idx_row.size()][idx_col.size()][k].p + S[idx_row.size()][idx_col.size()][k + 1].eta;

  for (auto&& i : reverse(idx_row)) {
    for (auto&& j : reverse(idx_col)) {
      S[i][j][idx_dep.size()].eta = S[i][j][idx_dep.size()].p + S[i + 1][j][idx_dep.size()].eta + S[i][j + 1][idx_dep.size()].eta - S[i + 1][j + 1][idx_dep.size()].eta;
    }
  }
  for (auto&& j : reverse(idx_col)) {
    for (auto&& k : reverse(idx_dep)) {
      S[idx_row.size()][j][k].eta = S[idx_row.size()][j][k].p + S[idx_row.size()][j + 1][k].eta + S[idx_row.size()][j][k + 1].eta - S[idx_row.size()][j + 1][k + 1].eta;
    }
  }
  for (auto&& k : reverse(idx_dep)) {
    for (auto&& i : reverse(idx_row)) {
      S[i][idx_col.size()][k].eta = S[i][idx_col.size()][k].p + S[i + 1][idx_col.size()][k].eta + S[i][idx_col.size()][k + 1].eta - S[i + 1][idx_col.size()][k + 1].eta;
    }
  }

  for (auto&& i : reverse(idx_row)) {
    for (auto&& j : reverse(idx_col)) {
      for (auto&& k : reverse(idx_dep)) {
	S[i][j][k].eta = S[i][j][k].p + S[i + 1][j][k].eta + S[i][j + 1][k].eta + S[i][j][k + 1].eta - S[i + 1][j + 1][k].eta - S[i + 1][j][k + 1].eta - S[i][j + 1][k + 1].eta + S[i + 1][j + 1][k + 1].eta;
      }
    }
  }
}
// e-projection
void computeP(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	if (S[i][j][k].p > EPSILON) {
	  double theta_sum = 0.0;
	  double theta_sum_prev = 0.0;
	  for (auto&& b : beta) {
	    if (get<0>(b.first) <= i && get<1>(b.first) <= j && get<2>(b.first) <= k) {
	      theta_sum += S[get<0>(b.first)][get<1>(b.first)][get<2>(b.first)].theta;
	      theta_sum_prev += S[get<0>(b.first)][get<1>(b.first)][get<2>(b.first)].theta_prev;
	    }
	  }
	  S[i][j][k].theta_sum = theta_sum;
	  S[i][j][k].theta_sum_prev = theta_sum_prev;
	  S[i][j][k].p = exp(theta_sum);
	}
      }
    }
  }
}
void renormalize(Poset& S) {
  // total sum
  double p_sum = 0.0;
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) p_sum += x.p;
      }
    }
  }
  // store the previous theta
  S[0][0][0].theta_prev = S[0][0][0].theta;
  // update theta(\bot)
  S[0][0][0].theta = S[0][0][0].theta_prev - log(p_sum);
  // update p
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) x.p *= exp(S[0][0][0].theta - S[0][0][0].theta_prev);
      }
    }
  }
}
void initialize(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  double size = 0.0;
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	if (S[i][j][k].nonzero) size += 1.0;
      }
    }
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	S[i][j][k].p = S[i][j][k].nonzero ? 1.0 / size : 0.0;
	S[i][j][k].theta = 0;
	S[i][j][k].theta_prev = 0;
      }
    }
  }
  S[0][0][0].theta = log(S[0][0][0].p);
  S[0][0][0].theta_prev = log(S[0][0][0].p);
  computeEta(S);
}

// ====================================== //
// ========== Natural gradient ========== //
// ====================================== //
void eProject(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  Int S_size = 0;
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) S_size++;
      }
    }
  }
  VectorXd theta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec = VectorXd::Zero(beta.size());
  for (Int i = 0; i < beta.size(); i++) {
    theta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    eta_vec[i]   = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second;
  }
  MatrixXd J(beta.size(), beta.size()); // Jacobian matrix
  for (Int i1 = 0; i1 < beta.size(); i1++) {
    for (Int i2 = 0; i2 < beta.size(); i2++) {
      Int i1_i = get<0>(beta[i1].first);
      Int i1_j = get<1>(beta[i1].first);
      Int i1_k = get<2>(beta[i1].first);
      Int i2_i = get<0>(beta[i2].first);
      Int i2_j = get<1>(beta[i2].first);
      Int i2_k = get<2>(beta[i2].first);
      J(i1, i2) = S[max(i1_i, i2_i)][max(i1_j, i2_j)][max(i1_k, i2_k)].eta;
      J(i1, i2) -= S[i1_i][i1_j][i1_k].eta * S[i2_i][i2_j][i2_k].eta;
    }
  }

  theta_vec += ((-1 * J).colPivHouseholderQr().solve(eta_vec));
  // theta_vec += (-1 * J).fullPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivLu().solve(eta_vec);

  // store theta
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta_prev = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta = theta_vec[i];
  }
  // update p
  computeP(S, beta);
  renormalize(S);
  computeEta(S);
}
// ====================================== //
// ========== Gradient descent ========== //
// ====================================== //
void grad(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  double eps = 0.1;
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta += eps * (beta[i].second - S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta);
    // update p
    computeP(S, beta);
    renormalize(S);
    computeEta(S);
  }
}

// compute the residual
double computeResidual(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  double res = 0.0;
  for (Int i = 0; i < beta.size(); i++) {
    res += pow(S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second, 2.0);
  }
  return sqrt(res);    
}
double computeRMSE(Tensor& X, Tensor& Y) {
  double rmse = 0.0;
  for (Int i = 0; i < X.size(); ++i) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int k = 0; k < X.front().front().size(); ++k) {
	rmse += pow(X[i][j][k] - Y[i][j][k], 2.0);
      }
    }
  }
  rmse /= (double)X.size() * (double)X.front().size() * (double)X.front().front().size();
  rmse = sqrt(rmse);
  return rmse;
}

// the main function for Legendre decomposition by natural gradient
double LegendreDecomposition(Tensor& X, Int core_size, double error_tol, double rep_max, bool verbose, Int type, Int const_type, Int *num_param) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();
  clock_t ts, te;
  // preprocess
  double X_sum = preprocess(X);
  // make a node matrix
  Poset S = Poset(n1, vector<vector<node>>(n2, vector<node>(n3)));
  makePosetTensor(X, S);
  computeEta(S);

  vector<pair<tuple<Int, Int, Int>, double>> beta;
  prepareForBeta(S);
  if (const_type == 1) {
    makeBetaNorm(S, beta, core_size);
    makeBetaCore(S, beta, core_size, true);
  } else if (const_type == 2) {
    makeBetaCore(S, beta, core_size, false);
  } else {
    makeBetaCore(S, beta, core_size, false);
  }
  initialize(S);
  cout << "  Number of parameters: " << beta.size() << endl << flush;
  *num_param = beta.size();

  // run Legendre deomposition
  if (verbose) cout << "----- Start Legendre decomposition -----" << endl << flush;
  double res = 0.0;
  double step = 1.0;
  Int exponent = 0;
  auto t_start = system_clock::now();
  while (step <= rep_max) {

    if (type == 1) {
      eProject(S, beta); // perform e-projection
    } else if (type == 2) {
      grad(S, beta); // perform gradient descent
    } else {
      eProject(S, beta); // perform e-projection
    }

    double res_prev = res;
    res = computeResidual(S, beta);
    if (res_prev >= EPSILON && res > res_prev) {
      cout << "  Terminate with current residual = " << res << endl;
      return step;
    }

    // output the residual
    if (verbose) {
      cout << "Step\t" << step << "\t" << "Residual\t" << res << endl << flush;
    } else {
      if (res < pow(10, -1.0 * (double)exponent)) {
	cout << "  Step "; if (step < 10.0) cout << " ";
	cout << step << ", Residual: " << res << endl;
	exponent++;
      }
    }
    if (res < error_tol) break;
    step += 1.0;
  }
  if (verbose) cout << "----- End   Legendre decomposition -----" << endl;

  // put results to X
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	X[i][j][k] = S[i][j][k].p * X_sum;
      }
    }
  }

  return step;
}
