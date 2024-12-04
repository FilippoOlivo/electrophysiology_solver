#include "ionic.hpp"#include <typeinfo>
#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>
#include <fstream>
#include <iostream>
#include <vector>
void save_vector_as_binary(const std::vector<double> &vec, const std::string &filename)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      throw std::runtime_error("Error opening file for writing.");
    }

  // Write the size of the vector first (optional but useful for loading)
  std::size_t size = vec.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));

  // Write the data
  file.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(double));

  file.close();
  std::cout << "Vector saved to " << filename << " in binary format." << std::endl;
}

BuenoOrovio::BuenoOrovio(const Parameters &params)
  : params(params)
{}

void
BuenoOrovio::setup(const IndexSet &locally_owned_dofs,
                   const IndexSet &locally_relevant_dofs,
                   const double   &dt)
{
  TimerOutput::Scope t(timer, "Setup ionic model");

  this->locally_owned_dofs    = locally_owned_dofs;
  this->locally_relevant_dofs = locally_relevant_dofs;
  this->dt                    = dt;

  for (unsigned int i = 0; i < N_VARS; ++i)
    {
      w_old[i].reinit(locally_owned_dofs, mpi_comm);
      w[i].reinit(locally_owned_dofs, mpi_comm);
      w_old[i] = 1.;
      w[i]     = w_old[i];
    }

  Iion.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  Iion = 0;
}


std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::alpha(const double u) const
{
  std::array<double, N_VARS> a;

  a[0] =
    (1.0 - utils::heaviside_sharp(u, params.V1)) /
    (utils::heaviside_sharp(u, params.V1m) * (params.tau1pp - params.tau1p) +
     params.tau1p);
  a[1] = (1.0 - utils::heaviside_sharp(u, params.V2)) /
         (utils::heaviside(u, params.V2m, params.k2) *
            (params.tau2pp - params.tau2p) +
          params.tau2p);
  a[2] = 1.0 / (utils::heaviside_sharp(u, params.V2) *
                  (params.tau3pp - params.tau3p) +
                params.tau3p);

  return a;
}



std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::beta(const double u) const
{
  std::array<double, N_VARS> b;

  b[0] = -utils::heaviside_sharp(u, params.V1) / params.tau1plus;
  b[1] = -utils::heaviside_sharp(u, params.V2) / params.tau2plus;
  b[2] = 0;

  return b;
}

std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::w_inf(const double u) const
{
  std::array<double, N_VARS> wi;

  wi[0] = 1.0 - utils::heaviside_sharp(u, params.V1m);
  wi[1] = utils::heaviside_sharp(u, params.Vo) *
            (params.w_star_inf - 1.0 + u / params.tau2inf) +
          1.0 - u / params.tau2inf;
  wi[2] = utils::heaviside(u, params.V3, params.k3);

  return wi;
}

double
BuenoOrovio::Iion_0d(const double                                   u_old,
                     const std::array<double, BuenoOrovio::N_VARS> &w) const
{
  TimerOutput::Scope t(timer, "Compute Iion");

  const double Iion_val =
    utils::heaviside_sharp(u_old, params.V1) * (u_old - params.V1) *
      (params.Vhat - u_old) * w[0] / params.taufi -
    (1.0 - utils::heaviside_sharp(u_old, params.V2)) * (u_old - 0.) /
      (utils::heaviside_sharp(u_old, params.Vo) *
         (params.tauopp - params.tauop) +
       params.tauop) -
    utils::heaviside_sharp(u_old, params.V2) /
      (utils::heaviside(u_old, params.Vso, params.kso) *
         (params.tausopp - params.tausop) +
       params.tausop) +
    utils::heaviside_sharp(u_old, params.V2) * w[1] * w[2] / params.tausi;

  return -Iion_val;
}

std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::solve_0d(const double                                   u_old,
                      const std::array<double, BuenoOrovio::N_VARS> &w) const
{
  TimerOutput::Scope t(timer, "Compute w");

  std::array<double, N_VARS> w_new;

  std::array<double, 3> a      = alpha(u_old);
  std::array<double, 3> b      = beta(u_old);
  std::array<double, 3> w_infs = w_inf(u_old);

  for (unsigned int i = 0; i < N_VARS; ++i)
    {
      w_new[i] = w[i] + dt * ((b[i] - a[i]) * w[i] + a[i] * w_infs[i]);
    }

  return w_new;
}

void
BuenoOrovio::solve(const LinearAlgebra::distributed::Vector<double> &u_old,
                   const double time)
{
  TimerOutput::Scope t(timer, "Update w and Iion at DoFs");

  // update w from t_n to t_{n+1} on the locally owned DoFs for all w's
  // On top of that, evaluate Iion at DoFs
  Iion.zero_out_ghost_values();
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      std::array<double, N_VARS> w_new =
        solve_0d(u_old[idx], {{w_old[0][idx], w_old[1][idx], w_old[2][idx]}});

      for (unsigned int i = 0; i < N_VARS; ++i)
        {
          w[i][idx] = w_new[i];
        }

      Iion[idx] = Iion_0d(u_old[idx], {{w[0][idx], w[1][idx], w[2][idx]}});
    }

  std::vector<double> local_vector(w[0].size() * 3);
  for (unsigned int i=0; i<3; i++)
    {
      unsigned int offset = w[0].size() * i;
      const auto &locally_owned = w[i].locally_owned_elements();
      for (unsigned int idx : locally_owned){
        local_vector[idx + offset] = w[i][idx];
      }
    }
  save_vector_as_binary(local_vector, "test.bin");
  // Write the distributed vector to the file

  Iion.update_ghost_values();

  w_old = w;
}



// Save a vector to a binary file
