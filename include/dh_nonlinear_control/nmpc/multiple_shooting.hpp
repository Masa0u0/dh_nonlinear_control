#pragma once

#include <vector>
#include <Eigen/Core>
#include <casadi/casadi.hpp>

namespace ctrl
{
class NMPC_MultipleShooting
{
public:
  NMPC_MultipleShooting(
    const std::function<casadi::MX(const casadi::MX&, const casadi::MX&)>& discreteDynamics,
    const std::function<casadi::MX(const casadi::MX&, const casadi::MX&)>& stepCost,
    const std::function<void(casadi::Opti&, const casadi::MX&)>& stateConstraints,
    const std::function<void(casadi::Opti&, const casadi::MX&)>& inputConstraints,
    const unsigned int& x_dim,
    const unsigned int& u_dim,
    const unsigned int& Hp,
    const unsigned int& Hu);

  Eigen::VectorXd step(const Eigen::VectorXd& x0, const Eigen::VectorXd& s);
  void setInitialGuess(const Eigen::VectorXd& stable_x, const Eigen::VectorXd& stable_u);

private:
  const std::function<casadi::MX(const casadi::MX&, const casadi::MX&)>& discreteDynamics_;
  const std::function<casadi::MX(const casadi::MX&, const casadi::MX&)>& stepCost_;
  const std::function<void(casadi::Opti&, const casadi::MX&)>& stateConstraints_;
  const std::function<void(casadi::Opti&, const casadi::MX&)>& inputConstraints_;
  const unsigned int x_dim_;
  const unsigned int u_dim_;
  const unsigned int Hp_;
  const unsigned int Hu_;

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
};
}  // namespace ctrl
