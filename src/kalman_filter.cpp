#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  VectorXd y = z - H_ * x_;
  x_ = x_ + (K * y);
  auto I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  //now we need y

  //predicted state in polar
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  //division by 0
  float eps = 0.0000001;
  if (std::abs(px) < eps) {
    px = eps;
    if (std::abs(py) < eps) {
      py = eps;
    }
  }

  float rho = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
  float rho_dot = (px * vx + py * vy) / rho;
  float theta = std::atan2(py, px);

  Vector3d h;
  h << rho, theta, rho_dot;

  VectorXd y = z - h;

  y[1] -= (2 * M_PI) * std::floor((y[1] + M_PI) / (2 * M_PI));

  //now that we have y, we can keep calculating kalman as usual

  x_ = x_ + (K * y);
  auto I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
