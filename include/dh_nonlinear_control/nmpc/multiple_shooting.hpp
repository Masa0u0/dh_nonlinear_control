#pragma once

#include <vector>
#include <Eigen/Core>
#include <casadi/casadi.hpp>

namespace ctrl
{
class NMPC_MultipleShootingModel
{
public:
  virtual casadi::MX discreteDynamics(const casadi::MX& x, const casadi::MX& u) = 0;

  virtual casadi::MX
  stepCost(const casadi::MX& x, const casadi::MX& u, const casadi::DM& x_ref) = 0;

  virtual void stateConstraints(casadi::Opti& opti, const casadi::MX& x) = 0;

  virtual void inputConstraints(casadi::Opti& opti, const casadi::MX& u) = 0;
};

class NMPC_MultipleShooting
{
public:
  NMPC_MultipleShooting(
    NMPC_MultipleShootingModel& model,
    const std::vector<double> decay_time_consts,
    const unsigned int& x_dim,
    const unsigned int& u_dim,
    const unsigned int& Hp,
    const unsigned int& Hu,
    const double& dt);

  Eigen::VectorXd step(const Eigen::VectorXd& x0, const Eigen::VectorXd& s);
  void setInitialGuess(const Eigen::VectorXd& stable_x, const Eigen::VectorXd& stable_u);

private:
  NMPC_MultipleShootingModel& model_;
  const casadi::DM decay_time_consts_;
  const unsigned int x_dim_;
  const unsigned int u_dim_;
  const unsigned int Hp_;
  const unsigned int Hu_;
  const double dt_;

  const casadi::Slice all_;
  casadi::DM x0_;
  casadi::DM s_;
  casadi::DM prev_xs_;
  casadi::DM prev_us_;
  Eigen::VectorXd opt_u_;

  /* 前回の解を初期推定解に設定する． */
  void setInitialGuess(casadi::Opti& opti, const casadi::MX& xs, const casadi::MX& us);
  void setObjective(casadi::Opti& opti, const casadi::MX& xs, const casadi::MX& us);
  void setConstraints(casadi::Opti& opti, const casadi::MX& xs, const casadi::MX& us);
  casadi::DM calcReferenceState(unsigned int k);
};
}  // namespace ctrl
