#include <dh_casadi_tools/conversion/casadi_eigen.hpp>

#include "../../../include/dh_nonlinear_control/nmpc/multiple_shooting.hpp"
#include "../../../include/dh_nonlinear_control/c2d.hpp"

using namespace std;
using namespace Eigen;
using namespace casadi;

namespace ctrl
{
NMPC_MultipleShooting::NMPC_MultipleShooting(
  const function<MX(const MX&, const MX&)>& discreteDynamics,
  const function<MX(const MX&, const MX&)>& stepCost,
  const function<void(Opti&, const MX&)>& stateConstraints,
  const function<void(Opti&, const MX&)>& inputConstraints,
  const unsigned int& x_dim,
  const unsigned int& u_dim,
  const unsigned int& Hp,
  const unsigned int& Hu)
  : discreteDynamics_(discreteDynamics),
    stepCost_(stepCost),
    stateConstraints_(stateConstraints),
    inputConstraints_(inputConstraints),
    x_dim_(x_dim),
    u_dim_(u_dim),
    Hp_(Hp),
    Hu_(Hu),
    x0_(x_dim, 1),
    s_(x_dim, 1),
    prev_xs_(DM::zeros(x_dim, Hp + 1)),
    prev_us_(DM::zeros(u_dim, Hu)),
    opt_u_(u_dim)
{
  assert(x_dim_ > 0);
  assert(u_dim_ > 0);
  assert(0 < Hu && Hu <= Hp);
}

VectorXd NMPC_MultipleShooting::step(const VectorXd& x0, const VectorXd& s)
{
  assert(x0.rows() == x_dim_);
  assert(s.rows() == x_dim_);

  tf::matrixEigenToCasadi(x0, x0_);
  tf::matrixEigenToCasadi(s, s_);

  Opti opti;
  auto xs = opti.variable(x_dim_, Hp_ + 1);  // x(0), ... , x(Hp)
  auto us = opti.variable(u_dim_, Hu_);      // u(0), ... , u(Hu - 1)

  setInitialGuess(opti, xs, us);
  setObjective(opti, xs, us);
  setConstraints(opti, xs, us);

  auto sol = opti.solve();
  prev_xs_ = sol.value(xs);
  prev_us_ = sol.value(us);

  tf::matrixCasadiToEigen(prev_us_(all_, 0), opt_u_);
  return opt_u_;
}

void NMPC_MultipleShooting::setInitialGuess(const VectorXd& stable_x, const VectorXd& stable_u)
{
  assert(stable_x.rows() == x_dim_);
  assert(stable_u.rows() == u_dim_);

  DM stable_x_dm(x_dim_, 1);
  DM stable_u_dm(u_dim_, 1);
  tf::matrixEigenToCasadi(stable_x, stable_x_dm);
  tf::matrixEigenToCasadi(stable_u, stable_u_dm);

  for (int k = 0; k <= Hp_; ++k)
  {
    prev_xs_(all_, k) = stable_x_dm;
  }
  for (int k = 0; k < Hu_; ++k)
  {
    prev_us_(all_, k) = stable_u_dm;
  }
}

void NMPC_MultipleShooting::setInitialGuess(Opti& opti, const MX& xs, const MX& us)
{
  opti.set_initial(xs, prev_xs_);
  opti.set_initial(us, prev_us_);
}

void NMPC_MultipleShooting::setObjective(Opti& opti, const MX& xs, const MX& us)
{
  MX obj = 0;

  for (int k = 0; k <= Hp_; ++k)
  {
    const auto& u = (k < Hu_) ? us(all_, k) : us(all_, Hu_ - 1);
    obj += stepCost_(xs(all_, k), u);
  }

  opti.minimize(obj);
}

void NMPC_MultipleShooting::setConstraints(Opti& opti, const MX& xs, const MX& us)
{
  // ダイナミクス
  opti.subject_to(xs(all_, 0) == x0_);
  for (int k = 0; k < Hp_; ++k)
  {
    const auto& u = (k < Hu_) ? us(all_, k) : us(all_, Hu_ - 1);
    opti.subject_to(xs(all_, k + 1) == discreteDynamics_(xs(all_, k), u));
  }

  // ユーザ制約
  for (int k = 1; k <= Hp_; ++k)
  {
    stateConstraints_(opti, xs(all_, k));
  }
  for (int k = 0; k < Hu_; ++k)
  {
    inputConstraints_(opti, us(all_, k));
  }
}
}  // namespace ctrl
