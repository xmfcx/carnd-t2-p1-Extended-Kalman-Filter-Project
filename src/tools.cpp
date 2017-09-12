#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size() || estimations.empty()) {
    cerr << "Error: Estimation size != GroundTruth size" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //mean
  rmse = rmse / estimations.size();

  //squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  MatrixXd Hj(3, 4);

  float px = (float) x_state(0);
  float py = (float) x_state(1);
  float vx = (float) x_state(2);
  float vy = (float) x_state(3);

  //check division by zero
  float c1 = px * px + py * py;
  if (c1 < 0.0001) {
    cerr << "Error: Can't do Jacobian, div by 0" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  Hj << px / c2, py / c2, 0, 0,
      -py / c1, px / c1, 0, 0,
      py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, px / c2,
      py / c2;


  return Hj;
}
