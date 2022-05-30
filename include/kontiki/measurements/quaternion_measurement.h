#pragma once
#ifndef KONTIKIV2_QUATERNION_MEASUREMENT_H
#define KONTIKIV2_QUATERNION_MEASUREMENT_H

#include <Eigen/Geometry>

#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>

namespace kontiki {
namespace measurements {

class QuaternionMeasurement {
  using Quaternion = Eigen::Quaterniond;
 public:
  QuaternionMeasurement(double t, const Quaternion &q, double weight=1.0, double coster_th=7.8):
    t_(t), q_(q), weight_(weight), huber_coster_th_(coster_th), huber_coster_(coster_th>=0.0?coster_th:1e38) {}
  QuaternionMeasurement(double t, const Eigen::Vector4d &qvec, double weight=1.0, double coster_th=7.8)
    : t_(t), q_(Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3))), weight_(weight), huber_coster_th_(coster_th), huber_coster_(coster_th>=0.0?coster_th:1e38) {}

  template<typename TrajectoryModel, typename T>
  Eigen::Quaternion<T> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return trajectory.Orientation(T(t_));
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    Eigen::Quaternion<T> qhat = Measure<TrajectoryModel, T>(trajectory);
    return T(weight_) * ErrorRaw<TrajectoryModel, T>(trajectory);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    Eigen::Quaternion<T> qhat = Measure<TrajectoryModel, T>(trajectory);
    Eigen::AngleAxis<T> aa { qhat.conjugate() * q_.cast<T>() };
    return aa.angle() * aa.axis();
  }

  // Measurement data
  double t_;
  Quaternion q_;
  double weight_;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const QuaternionMeasurement &m) : measurement(m) {}

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      auto trajectory = entity::Map<TrajectoryModel, T>(params, meta);
      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory);
      return true;
    }

    const QuaternionMeasurement& measurement;
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
} // namespace measurements
} // namespace kontiki


#endif //KONTIKIV2_QUATERNION_MEASUREMENT_H
