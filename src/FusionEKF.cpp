#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;


  //state covariance matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  //the initial transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!is_initialized_) {
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      double px = rho * std::cos(phi);
      double py = rho * std::sin(phi);
      double vx = 0;
      double vy = 0;

      ekf_.x_ << px, py, vx, vy;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_
          << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  //prediction time!!

  float dt = (float) (measurement_pack.timestamp_ - previous_timestamp_) /
             (float) 1000000;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

  float dt2 = dt * dt;
  float dt3 = dt2 * dt / 2;
  float dt4 = dt3 * dt / 2;

  float noise_ax = 9;
  float noise_ay = 9;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 * noise_ax, 0, dt3 * noise_ax, 0,
      0, dt4 * noise_ay, 0, dt3 * noise_ay,
      dt3 * noise_ax, 0, dt2 * noise_ax, 0,
      0, dt3 * noise_ay, 0, dt2 * noise_ay;

  if (dt >= 0.000001) {
    ekf_.Predict();
  }

  //update time!!

  VectorXd x_new = measurement_pack.raw_measurements_;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    if ((x_new(0) != 0) || (x_new(1) != 0)) {
      ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.R_ = R_radar_;
      ekf_.UpdateEKF(x_new);
    }
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(x_new);
  }

  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
