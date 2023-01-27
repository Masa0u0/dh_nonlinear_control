#pragma once

#include <cassert>
#include <casadi/casadi.hpp>

namespace ctrl
{
template <typename T>
T c2d_euler(const std::function<T(const T&)>& f, const T& x, const double& dt)
{
  assert(dt >= 0.);

  return x + f(x) * dt;
}

template <typename T>
T c2d_euler(const std::function<T(const T&, const T&)>& f, const T& x, const T& u, const double& dt)
{
  assert(dt >= 0.);

  return x + f(x, u) * dt;
}

template <typename T>
T c2d_rk4(const std::function<T(const T&)>& f, const T& x, const double& dt)
{
  assert(dt >= 0.);

  T k1 = f(x);
  T k2 = f(x + (dt / 2) * k1);
  T k3 = f(x + (dt / 2) * k2);
  T k4 = f(x + dt * k3);

  return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

template <typename T>
T c2d_rk4(const std::function<T(const T&, const T&)>& f, const T& x, const T& u, const double& dt)
{
  assert(dt >= 0.);

  // uはdt間で一定値をとると仮定
  T k1 = f(x, u);
  T k2 = f(x + (dt / 2) * k1, u);
  T k3 = f(x + (dt / 2) * k2, u);
  T k4 = f(x + dt * k3, u);

  return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}
}  // namespace ctrl
