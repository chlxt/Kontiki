
#pragma once
#ifndef KONTIKIV2_LIDAR_POSITION_MEASUREMENT_pairwise_H
#define KONTIKIV2_LIDAR_POSITION_MEASUREMENT_pairwise_H

#include <Eigen/Dense>

#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>
#include "../sensors/lidar.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPositionMeasurementPairwise {
  using Vector3 = Eigen::Matrix<double, 3, 1>;
  using Quaternion = Eigen::Quaternion<double>;
 public:
  LiDARPositionMeasurementPairwise(std::shared_ptr<LiDARModel> lidar,
    double tc, const Vector3& pc, const Quaternion& qc,
    double t0, const Vector3& p0,
    double weight=1.0, double coster_th=7.8)
    : lidar_(lidar),
      t_(tc), p_(pc), q_(qc),
      t0_(t0), p0_(p0),
      weight_(weight), huber_coster_th_(coster_th), huber_coster_(coster_th>=0.0?coster_th:1e38)
    {}

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory,
                                 const type::LiDAR<LiDARModel, T> &lidar) const {
    auto q_W_Bc = trajectory.Evaluate(T(t_), trajectories::EvaluationFlags::EvalOrientation)->orientation;
    auto q_W_B0 = trajectory.Evaluate(T(t0_), trajectories::EvaluationFlags::EvalOrientation)->orientation;
    auto p_W_Bc = trajectory.Evaluate(T(t_), trajectories::EvaluationFlags::EvalPosition)->position;
    auto p_W_B0 = trajectory.Evaluate(T(t0_), trajectories::EvaluationFlags::EvalPosition)->position;

    const Eigen::Quaternion<T> q_B_L = lidar.relative_orientation();
    const Eigen::Matrix<T, 3, 1> p_B_L = lidar.relative_position();
    const Eigen::Quaternion<T> q_L_B = q_B_L.conjugate();
    const Eigen::Matrix<T, 3, 1> p_L_B = q_B_L.conjugate() * (-p_B_L);

    const Eigen::Matrix<T, 3, 1> p_W = q_W_B0 * p_B_L + p_W_B0;
    const Eigen::Matrix<T, 3, 1> p_W_Lc = q_W_Bc * p_B_L + p_W_Bc;
    const Eigen::Quaternion<T> q_W_Lc = q_W_Bc * q_B_L;
    Eigen::Quaternion<T> qc = q_.cast<T>();
    auto pc = p_;
    const Eigen::Matrix<T, 3, 1> p_0 = qc * q_W_Lc.conjugate() * (p_W - p_W_Lc) + pc;

    return p_0;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    return T(weight_) * ErrorRaw<TrajectoryModel, T>(trajectory, lidar);
  }

  template<typename TrajectoryModel>
  Eigen::Matrix<double, 3, 1> Error(const type::Trajectory<TrajectoryModel, double> &trajectory) const {
    return Error<TrajectoryModel>(trajectory, *lidar_);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    return p0_ - Measure<TrajectoryModel, T>(trajectory, lidar);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<double, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return ErrorRaw<TrajectoryModel, T>(trajectory, *lidar_);
  }

  // Measurement data
  std::shared_ptr<LiDARModel> lidar_;
  double t_, t0_;
  Vector3 p_;
  Quaternion q_;
  Vector3 p0_;
  double weight_;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPositionMeasurementPairwise<LiDARModel> &m) : measurement(m) {}

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

    const LiDARPositionMeasurementPairwise& measurement;
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
                                         huber_coster_th_ < 0.0 ? nullptr : &huber_coster_,
                                         entity::ParameterInfo<double>::ToParameterBlocks(parameters));
  }

  ceres::HuberLoss huber_coster_;
  double huber_coster_th_ = 7.8;

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};


} // namespace measurements
} // namespace kontiki


#endif
