//
// Created by hannes on 2017-11-29.
//

#ifndef KONTIKIV2_POSITION_MEASUREMENT_H
#define KONTIKIV2_POSITION_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>

namespace kontiki {
namespace measurements {

class PositionMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  PositionMeasurement(double t, const Vector3 &p, double weight=1.0, double coster_th=7.8):
    t_(t), p_(p), weight_(weight), off_(0.0, 0.0, 0.0), huber_coster_th_(coster_th), huber_coster_(coster_th>=0.0?coster_th:1e38) {};

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    if(off_.cwiseAbs().sum() < 1e-8) {
      return trajectory.Position(T(t_));
    } else {
      auto orien_WB = trajectory.Orientation(T(t_));
      return trajectory.Position(T(t_)) + orien_WB * off_.cast<T>();
    }
  };

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return T(weight_) * (p_.cast<T>() - Measure<TrajectoryModel, T>(trajectory));
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return p_.cast<T>() - Measure<TrajectoryModel, T>(trajectory);
  }

  // Measurement data
  double t_;
  Vector3 p_;
  double weight_;
  Vector3 off_;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const PositionMeasurement &m) : measurement(m) {};

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      auto trajectory = entity::Map<TrajectoryModel, T>(params, meta);
      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory);
      return true;
    }

    const PositionMeasurement& measurement;
    typename TrajectoryModel::Meta meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameter_info;

    // Add trajectory to problem
    //estimator.trajectory()->AddToProblem(estimator.problem(), residual->meta, parameter_blocks, parameter_sizes);
    estimator.AddTrajectoryForTimes({{t_,t_}}, residual->meta, parameter_info);
    for (auto& pi : parameter_info) {
      cost_function->AddParameterBlock(pi.size);
    }

    // Add measurement
    cost_function->SetNumResiduals(3);
    // If we had any measurement parameters to set, this would be the place

    // Give residual block to estimator problem
    estimator.problem().AddResidualBlock(cost_function,
                                         huber_coster_th_ < 0.0 ? nullptr : &huber_coster_,
                                         entity::ParameterInfo<double>::ToParameterBlocks(parameter_info));
  }

  ceres::HuberLoss huber_coster_;
  double huber_coster_th_ = 7.8;

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

class AnotherMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  AnotherMeasurement(double t, const Vector3 &p) : t_(t), p_(p) {};

  void hello() {
    std::cout << "Hello from AnotherMeasurement!" << std::endl;
  }

 protected:
  double t_;
  Vector3 p_;
};

} // namespace measurements
} // namespace kontiki


#endif //KONTIKIV2_POSITION_MEASUREMENT_H
