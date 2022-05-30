#pragma once
#ifndef KONTIKIV2_LIDAR_QUATERNION_MEASUREMENT_RELATIVE_H
#define KONTIKIV2_LIDAR_QUATERNION_MEASUREMENT_RELATIVE_H

#include <Eigen/Dense>

#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>
#include "../sensors/lidar.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARQuaternionMeasurementRelative {
  using Quaternion = Eigen::Quaternion<double>;
 public:
  LiDARQuaternionMeasurementRelative(std::shared_ptr<LiDARModel> lidar, double t, const Quaternion &q, double t0, double weight=1.0, double coster_th=7.8)
    : lidar_(lidar), t_(t), t0_(t0), q_(q), weight_(weight), huber_coster_(coster_th) {}

  template<typename TrajectoryModel, typename T>
  Eigen::Quaternion<T> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory,
                                 const type::LiDAR<LiDARModel, T> &lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition | trajectories::EvaluationFlags::EvalOrientation;
    auto T_M_I = trajectory.Evaluate(T(t_), flags);
    auto T_M_I0 = trajectory.Evaluate(T(t0_), flags);

    const Eigen::Quaternion<T> q_I_L = lidar.relative_orientation();
    const Eigen::Matrix<T, 3, 1> p_I_L = lidar.relative_position();
    const Eigen::Quaternion<T> q_L_I = q_I_L.conjugate();
    Eigen::Matrix<T, 3, 1> p_L_I = q_I_L.conjugate() * (-p_I_L);

    // const Eigen::Matrix<T, 3, 1> p_L = q_.cast<T>();
    const Eigen::Quaternion<T> q_I = q_I_L;
    const Eigen::Quaternion<T> q_M = T_M_I->orientation * q_I;
    const Eigen::Quaternion<T> q_I0 = T_M_I0->orientation.conjugate() * q_M;
    const Eigen::Quaternion<T> q_L0 = q_L_I * q_I0;

    return q_L0;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Quaternion<T> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return Measure<TrajectoryModel, T>(trajectory, *lidar_);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    return weight_ * ErrorRaw<TrajectoryModel, T>(trajectory, lidar);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<double, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return weight_ * ErrorRaw<TrajectoryModel, T>(trajectory, *lidar_);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Quaternion<T> qhat = Measure<TrajectoryModel, T>(trajectory, lidar);
    return 2.0 * (qhat.conjugate() * q_.cast<T>()).vec();
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<double, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return ErrorRaw<TrajectoryModel, T>(trajectory, *lidar_);
  }

  // Measurement data
  std::shared_ptr<LiDARModel> lidar_;
  double t_, t0_;
  Quaternion q_;
  double weight_;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARQuaternionMeasurementRelative<LiDARModel> &m) : measurement(m) {}

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);

      offset += trajectory_meta.NumParameters();
      auto lidar = entity::Map<LiDARModel, T>(&params[offset], lidar_meta);

      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory, lidar);
      return true;
    }

    const LiDARQuaternionMeasurementRelative& measurement;
    typename TrajectoryModel::Meta trajectory_meta;
    typename LiDARModel::Meta lidar_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameters;

    // Find timespans for the two observations
    double t_min, t_max;
    if(this->lidar_->TimeOffsetIsLocked()) {
        t_min = t_;
        t_max = t_;
    }
    else {
        t_min = t_ - this->lidar_->max_time_offset();
        t_max = t_ + this->lidar_->max_time_offset();
    }

    double t0_min, t0_max;
    if(this->lidar_->TimeOffsetIsLocked()) {
        t0_min = t0_;
        t0_max = t0_;
    }
    else {
        t0_min = t0_ - this->lidar_->max_time_offset();
        t0_max = t0_ + this->lidar_->max_time_offset();
    }

    // Add trajectory to problem
  if(t0_min <= t_min) {
    estimator.AddTrajectoryForTimes({
                                      {t0_min, t0_max},
                                      {t_min, t_max}
                                    },
                                    residual->trajectory_meta,
                                    parameters);

    lidar_->AddToProblem(estimator.problem(),
                          {
                            {t0_min, t0_max},
                            {t_min, t_max}
                          },
                          residual->lidar_meta, parameters);
    } else {
    estimator.AddTrajectoryForTimes({
                                      {t_min, t_max},
                                      {t0_min, t0_max}
                                    },
                                    residual->trajectory_meta,
                                    parameters);

    lidar_->AddToProblem(estimator.problem(),
                          {
                            {t_min, t_max},
                            {t0_min, t0_max}
                          },
                          residual->lidar_meta, parameters);
    }


    for (auto& pi : parameters) {
      cost_function->AddParameterBlock(pi.size);
    }

    // Add measurement
    cost_function->SetNumResiduals(3);
    // If we had any measurement parameters to set, this would be the place

    // Give residual block to estimator problem
    estimator.problem().AddResidualBlock(cost_function,
                                         &huber_coster_,
                                         entity::ParameterInfo<double>::ToParameterBlocks(parameters));
  }

  ceres::HuberLoss huber_coster_;

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};


} // namespace measurements
} // namespace kontiki


#endif
