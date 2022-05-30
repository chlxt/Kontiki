//
// Created by hannes on 2017-11-29.
//

#ifndef KONTIKIV2_LIDAR_POSITION_MEASUREMENT_H
#define KONTIKIV2_LIDAR_POSITION_MEASUREMENT_H

#include <Eigen/Dense>

#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>
#include "../sensors/lidar.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPositionMeasurement {
  using Vector3 = Eigen::Matrix<double, 3, 1>;
 public:
  LiDARPositionMeasurement(std::shared_ptr<LiDARModel> lidar, double t, const Vector3 &p, double weight=1.0, double coster_th=7.8)
    : lidar_(lidar), t_(t), p_(p), weight_(weight), huber_coster_th_(coster_th), huber_coster_(coster_th>=0.0?coster_th:1e38) {}

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory,
                                 const type::LiDAR<LiDARModel, T> &lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition | trajectories::EvaluationFlags::EvalOrientation;
    auto T_W_B = trajectory.Evaluate(T(t_), flags);

    const Eigen::Matrix<T, 3, 1> p_B_L = lidar.relative_position();
    const Eigen::Quaternion<T> q_B_L = lidar.relative_orientation();

    Eigen::Matrix<T, 3, 1> p_W_L = T_W_B->orientation * p_B_L + T_W_B->position;

    return p_W_L;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    return T(weight_) * (p_- Measure<TrajectoryModel, T>(trajectory, lidar));
  }

  template<typename TrajectoryModel>
  Eigen::Matrix<double, 3, 1> Error(const type::Trajectory<TrajectoryModel, double> &trajectory) const {
    return Error(trajectory, *lidar_);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    return p_ - Measure<TrajectoryModel, double>(trajectory, *lidar_);
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<double, 3, 1> ErrorRaw(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return ErrorRaw<TrajectoryModel, T>(trajectory, *lidar_);
  }

  // Measurement data
  std::shared_ptr<LiDARModel> lidar_;
  double t_;
  Vector3 p_;
  double weight_;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPositionMeasurement<LiDARModel> &m) : measurement(m) {}

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

    const LiDARPositionMeasurement& measurement;
    typename TrajectoryModel::Meta trajectory_meta;
    typename LiDARModel::Meta lidar_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameter_info;

    // Add trajectory to problem
    //estimator.trajectory()->AddToProblem(estimator.problem(), residual->meta, parameter_blocks, parameter_sizes);
    estimator.AddTrajectoryForTimes({{t_,t_}}, residual->trajectory_meta, parameter_info);
    lidar_->AddToProblem(estimator.problem(), {{t_,t_}}, residual->lidar_meta, parameter_info);


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


#endif //KONTIKIV2_LIDAR_POSITION_MEASUREMENT_H
