#include <deal.II/base/config.h>

#include "ionic_nn.hpp"

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "create_graph.hpp"
#include "torch_inference.hpp"


BuenoOrovioGNO::BuenoOrovioGNO(const Parameters &params)
  : params(params)
{}

void
BuenoOrovioGNO::setup(const IndexSet &locally_owned_dofs,
                      const IndexSet &locally_relevant_dofs,
                      DoFHandler<3>  &dof_handler,
                      MappingQ<3>    &mapping)
{
  TimerOutput::Scope t(timer, "Setup ionic model");

  GraphCreator graph_saver(mpi_rank, mpi_size, mpi_comm);

  auto [edge_index, edge_attr] =
    graph_saver.build_graph<3>(dof_handler, mapping);

  this->locally_owned_dofs    = locally_owned_dofs;
  this->locally_relevant_dofs = locally_relevant_dofs;

  Iion.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  Iion            = 0;
  torch_inference = TorchInference("../scripted_model_ref_w.pt",
                                   edge_index,
                                   edge_attr,
                                   dof_handler.n_locally_owned_dofs());
  w_tensor =
    torch::ones({dof_handler.n_locally_owned_dofs(), 3}, torch::kDouble);
  u_old_tensor =
    torch::ones({dof_handler.n_locally_owned_dofs(), 1}, torch::kDouble);
}

void
BuenoOrovioGNO::solve_w(const LinearAlgebra::distributed::Vector<double> &u_old)
{
  torch::NoGradGuard no_grad;

  torch_inference.run(w_tensor);
  auto w_data = w_tensor.accessor<double, 2>();
  Iion.zero_out_ghost_values();
  unsigned int i = 0;
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      Iion[idx] =
        Iion_0d(u_old[idx], {{w_data[i][0], w_data[i][1], w_data[i][2]}});
    }
  Iion.update_ghost_values();
}

void
BuenoOrovioGNO::solve_uw(
  const LinearAlgebra::distributed::Vector<double> &u_old)
{
  torch::NoGradGuard no_grad;
  unsigned int       i = 0;

  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      u_old_tensor[i][0] = u_old[idx];
      i++;
    }
  w_tensor = torch::cat({u_old_tensor, w_tensor}, -1);

  torch_inference.run(w_tensor);
  auto w_data = w_tensor.accessor<double, 2>();
  Iion.zero_out_ghost_values();
  i = 0;
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      Iion[idx] =
        Iion_0d(u_old[idx], {{w_data[i][0], w_data[i][1], w_data[i][2]}});
    }
  Iion.update_ghost_values();
}


void
BuenoOrovioGNO::solve_u(const LinearAlgebra::distributed::Vector<double> &u_old)
{
  torch::NoGradGuard no_grad;
  unsigned int       i = 0;

  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      u_old_tensor[i][0] = u_old[idx];
      i++;
    }

  torch_inference.run(w_tensor, u_old_tensor);
  auto w_data = w_tensor.accessor<double, 2>();
  Iion.zero_out_ghost_values();
  i = 0;
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      Iion[idx] =
        Iion_0d(u_old[idx], {{w_data[i][0], w_data[i][1], w_data[i][2]}});
    }
  Iion.update_ghost_values();
}

double
BuenoOrovioGNO::Iion_0d(const double                 u_old,
                        const std::array<double, 3> &w) const
{
  // TimerOutput::Scope t(timer, "Compute Iion");

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