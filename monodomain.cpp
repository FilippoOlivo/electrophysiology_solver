#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <cmath>
#include <fstream>
#include "applied_current.hpp"
#include "common.hpp"
#include "ionic.hpp"
#include "save_utils.hpp"

using namespace dealii;

class Monodomain : public Common
{
public:
  class Parameters : public ParameterAcceptor
  {
  public:
      unsigned int fe_degree  = 1;
    unsigned int map_degree = 1;

    double dt       = 1e-3;
    double time_end = 1.;

    double sigma = 1e-4;
    bool save_hdf5 = false;
    bool save_xdmf = false;
    bool save_vtu = true;

    
    Parameters()
      : ParameterAcceptor("Monodomain solver")
    {
      add_parameter("fe_degree", fe_degree, "Finite Element degree");
      add_parameter("map_degree", map_degree, "Mapping degree");
      add_parameter("map_degree", map_degree, "Mapping degree");
      add_parameter("dt", dt, "Time step");
      add_parameter("time_end", time_end, "Final time");
      add_parameter("sigma", sigma, "Conductivity");
      add_parameter("save_hdf5", save_hdf5, "Save in hdf5 format");
      add_parameter("save_xdmf", save_xdmf, "Save in xdmf format");
      add_parameter("save_vtu", save_vtu, "Save in vtu format");

    }

    unsigned int fe_degree  = 1;
    unsigned int map_degree = 0;

    double dt       = 1e-3;
    double time_end = 1.;
    unsigned int fe_degree  = 1;
    unsigned int map_degree = 0;

    double dt       = 1e-3;
    double time_end = 1.;

    double sigma = 1e-4;
    bool save_hdf5 = false;
    bool save_xdmf = false;
    bool save_vtu = true;
  };

  Monodomain(const Parameters                 &solver_params,
             const BuenoOrovio::Parameters    &ionic_model_params,
             const AppliedCurrent::Parameters &applied_current_params);
  Monodomain(const Parameters                 &solver_params,
             const BuenoOrovio::Parameters    &ionic_model_params,
             const AppliedCurrent::Parameters &applied_current_params);

  void
  run();

private:
  void
  setup();

  void
  assemble_time_independent_matrix();

  void
  assemble_time_terms();
  void
  solve();

  void
  output_results();

  const Parameters              &params;
  std::unique_ptr<BuenoOrovio>   ionic_model;
  std::unique_ptr<FEValues<dim>> fe_values;

  const Parameters              &params;
  std::unique_ptr<BuenoOrovio>   ionic_model;
  std::unique_ptr<FEValues<dim>> fe_values;

  std::unique_ptr<Function<dim>>                 Iapp;
  parallel::fullydistributed::Triangulation<dim> tria;
  MappingQ<dim>                                  mapping;
  FE_Q<dim>                                      fe;
  DoFHandler<dim>                                dof_handler;
  SparsityPattern                                sparsity;
  AffineConstraints<double>                      constraints;
  TrilinosWrappers::PreconditionAMG              amg_preconditioner;
  TrilinosWrappers::SparseMatrix                 mass_matrix_dt;
  TrilinosWrappers::SparseMatrix                 laplace_matrix;
  TrilinosWrappers::SparseMatrix                 system_matrix;
  LinearAlgebra::distributed::Vector<double>     system_rhs;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  LinearAlgebra::distributed::Vector<double> u_old;
  LinearAlgebra::distributed::Vector<double> u;

  //   Time stepping parameters
  double       time;
  const double dt;
  unsigned int time_step;
  const double time_end;
};



Monodomain::Monodomain(const Parameters                 &solver_params,
                       const BuenoOrovio::Parameters    &ionic_model_params,
                       const AppliedCurrent::Parameters &applied_current_params)
  : params(solver_params)
  , ionic_model(std::make_unique<BuenoOrovio>(ionic_model_params))
Monodomain::Monodomain(const Parameters                 &solver_params,
                       const BuenoOrovio::Parameters    &ionic_model_params,
                       const AppliedCurrent::Parameters &applied_current_params)
  : params(solver_params)
  , ionic_model(std::make_unique<BuenoOrovio>(ionic_model_params))
  , Iapp(std::make_unique<AppliedCurrent>(applied_current_params))
  , tria(mpi_comm)
  , mapping(params.map_degree)
  , fe(params.fe_degree)
  , dof_handler(tria)
  , time(0)
  , dt(params.dt)
  , time_step(0)
  , time_end(params.time_end)

{}



void
Monodomain::setup()
{
  TimerOutput::Scope t(timer, "Setup monodomain");
  fe_values =
    std::make_unique<FEValues<dim>>(mapping,
                                    fe,
                                    QGauss<dim>(fe.degree + 1),
                                    update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points);

  dof_handler.distribute_dofs(fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_comm,
                                             locally_relevant_dofs);

  mass_matrix_dt.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

  u_old.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  u.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);


  ionic_model->setup(locally_owned_dofs, locally_relevant_dofs, dt);
}


/*
 * Assemble the time independent block M/dt + A
 */
void
Monodomain::assemble_time_independent_matrix()
{
  TimerOutput::Scope t(timer, "Assemble time independent terms");

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_dt(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix         = 0;
          cell_mass_matrix_dt = 0;
          fe_values->reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += params.sigma *
                                           fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);

                      cell_mass_matrix_dt(i, j) +=
                        (1. / dt) * fe_values->shape_value(i, q_index) *
                        fe_values->shape_value(j, q_index) *
                        fe_values->JxW(q_index);
                    }
                }
            }

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 laplace_matrix);
          constraints.distribute_local_to_global(cell_mass_matrix_dt,
                                                 local_dof_indices,
                                                 mass_matrix_dt);
        }
    }
  mass_matrix_dt.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}



void
Monodomain::assemble_time_terms()
{
  TimerOutput::Scope t(timer, "Assemble time dependent terms");

  system_rhs = 0;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values->reinit(cell);

          const auto        &q_points  = fe_values->get_quadrature_points();
          const unsigned int n_qpoints = q_points.size();

          std::vector<double> applied_currents(n_qpoints);
          Iapp->value_list(q_points, applied_currents);

          std::vector<double> ion_at_qpoints(n_qpoints);
          fe_values->get_function_values(ionic_model->Iion, ion_at_qpoints);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) +=
                    (applied_currents[q_index] - ion_at_qpoints[q_index]) *
                    fe_values->shape_value(i, q_index) *
                    fe_values->JxW(q_index);
                }
            }

          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs);
        }
    }
  system_rhs.compress(VectorOperation::add);

  mass_matrix_dt.vmult_add(system_rhs,
                           u_old); // Add to system_rhs (M/dt) * u_n
}

void
Monodomain::solve()
{
  TimerOutput::Scope t(timer, "Solve");

  SolverControl solver_control(1000, 1e-10);

  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
  solver.solve(system_matrix, u, system_rhs, amg_preconditioner);

  constraints.distribute(u);

  pcout << "\tNumber of outer iterations: " << solver_control.last_step()
        << std::endl;
}

void
Monodomain::output_results()
{
  TimerOutput::Scope t(timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(u,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  //
  for (unsigned int i = 0; i < ionic_model->w.size(); ++i)
  for (unsigned int i = 0; i < ionic_model->w.size(); ++i)
    {
      data_out.add_data_vector(ionic_model->w[i],
      data_out.add_data_vector(ionic_model->w[i],
                               "w" + std::to_string(i),
                               DataOut<dim>::type_dof_data);
    }
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const bool export_mesh = (time_step == 0);

  const std::string basename    = "output";
  const std::string filename_h5 = basename + "_" + std::to_string(time) + ".h5";
  const std::string filename_xdmf =
    basename + "_" + std::to_string(time) + ".xdmf";
  const std::string filename_mesh =
    basename + "_" + std::to_string(0.0) + ".h5";
  const std::string filename_vtu =
  basename + "_" + std::to_string(time) + ".vtu";
  DataOutBase::DataOutFilter data_filter(
      DataOutBase::DataOutFilterFlags(true, true));

  if (params.save_hdf5){
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(
      data_filter, export_mesh, filename_mesh, filename_h5, mpi_comm);
  }
  if (params.save_xdmf){
    std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
      data_filter, filename_mesh, filename_h5, time, mpi_comm)});

    data_out.write_xdmf_file(xdmf_entries, filename_xdmf, mpi_comm);
  }
  if (params.save_vtu)
    data_out.write_vtu_in_parallel(filename_vtu, mpi_comm);
}

void
Monodomain::run()
{
  // Create mesh
  {
    TimerOutput::Scope t(timer, "Create mesh");
    Triangulation<dim> tria_dummy;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(tria_dummy);
    std::ifstream mesh_file("../idealized_lv.msh");
    grid_in.read_msh(mesh_file);

    const double scale_factor = 1e-3;
    GridTools::scale(scale_factor, tria_dummy);

    GridTools::partition_triangulation(mpi_size, tria_dummy);

    const TriangulationDescription::Description<dim, dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(tria_dummy, mpi_comm);

    tria.create_triangulation(description);
  }

  pcout << "\tNumber of active cells:       " << tria.n_global_active_cells()
        << std::endl;

  setup();
  save_dofs_location<dim>(dof_handler, locally_owned_dofs, mapping, mpi_rank, mpi_size, mpi_comm);
  pcout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  u_old = -84e-3;
  u     = u_old;
  assemble_time_independent_matrix();
  // M/dt + A
  system_matrix.copy_from(mass_matrix_dt);
  system_matrix.add(+1, laplace_matrix);
  amg_preconditioner.initialize(system_matrix);

  while (time <= time_end)
    {
      time += dt;
      Iapp->set_time(time);
      ionic_model->solve(u_old, time);
      assemble_time_terms();
      solve();
      pcout << "Solved at t = " << time << std::endl;
      ++time_step;
      if ((time_step % 10 == 0))
        output_results();
      u_old = u;
    }
  pcout << std::endl;
}

void
Monodomain::save_dofs_location()
{
  std::map<types::global_dof_index, Point<3>> locations = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  std::vector<Point<3>> local_locations(locally_owned_dofs.size());
  unsigned int i = 0;
  for (auto idx : locally_owned_dofs)
    {
      local_locations[i] = locations[idx];
      ++i;
    }
  std::vector<double> x(dof_handler.n_locally_owned_dofs());
  std::vector<double> y(dof_handler.n_locally_owned_dofs());
  std::vector<double> z(dof_handler.n_locally_owned_dofs());
  std::vector<int> local_index(dof_handler.n_locally_owned_dofs());

  for (unsigned int i=0; i<dof_handler.n_locally_owned_dofs(); i++)
    {
      x[i] = local_locations[i][0];
      y[i] = local_locations[i][1];
      z[i] = local_locations[i][2];
    }
  int local_size = dof_handler.n_locally_owned_dofs();

  std::vector<int> per_proc_size(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT, per_proc_size.data(), 1,
                MPI_INT, mpi_comm);

  std::vector<int> displacement(mpi_size, 0);
  for (unsigned int i = 1; i < mpi_size; i++)
    {
      displacement[i] = displacement[i - 1] + per_proc_size[i - 1];
    }

  std::vector<double> global_x;
  std::vector<double> global_y;
  std::vector<double> global_z;
  std::vector<int> global_index;
  if (mpi_rank == 0)
    {
      global_x.resize(dof_handler.n_dofs());
      global_y.resize(dof_handler.n_dofs());
      global_z.resize(dof_handler.n_dofs());
      global_index.resize(dof_handler.n_dofs());
    }

  MPI_Gatherv(x.data(), // send buffer
           local_size, // send size
           MPI_DOUBLE, // type
           mpi_rank == 0 ? global_x.data() : nullptr, // receive buffer
           per_proc_size.data(), // receive size
           displacement.data(),
           MPI_DOUBLE, // receive type
           0, // main processor
           mpi_comm // MPI Comminicator
           );
  MPI_Gatherv(y.data(), // send buffer
         local_size, // send size
         MPI_DOUBLE, // type
         mpi_rank == 0 ? global_x.data() : nullptr, // receive buffer
         per_proc_size.data(), // receive size
         displacement.data(),
         MPI_DOUBLE, // receive type
         0, // main processor
         mpi_comm // MPI Comminicator
         );
  MPI_Gatherv(z.data(), // send buffer
         local_size, // send size
         MPI_DOUBLE, // type
         mpi_rank == 0 ? global_x.data() : nullptr, // receive buffer
         per_proc_size.data(), // receive size
         displacement.data(),
         MPI_DOUBLE, // receive type
         0, // main processor
         mpi_comm // MPI Comminicator
         );
  MPI_Gatherv(local_index.data(), // send buffer
       local_size, // send size
       MPI_INT, // type
       mpi_rank == 0 ? global_index.data() : nullptr, // receive buffer
       per_proc_size.data(), // receive size
       displacement.data(),
       MPI_INT, // receive type
       0, // main processor
       mpi_comm // MPI Comminicator
       );
  if (mpi_rank == 0)
    {
    save_vector_as_binary(global_x, "global_x.bin", dof_handler.n_dofs());
    save_vector_as_binary(global_y, "global_y.bin", dof_handler.n_dofs());
    save_vector_as_binary(global_z, "global_z.bin", dof_handler.n_dofs());

    }
  MPI_Barrier(mpi_comm);
}

void
Monodomain::run()
{
  // Create mesh
  {
    TimerOutput::Scope t(timer, "Create mesh");
    Triangulation<dim> tria_dummy;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(tria_dummy);
    std::ifstream mesh_file("../idealized_lv.msh");
    grid_in.read_msh(mesh_file);

    const double scale_factor = 1e-3;
    GridTools::scale(scale_factor, tria_dummy);

    GridTools::partition_triangulation(mpi_size, tria_dummy);

    const TriangulationDescription::Description<dim, dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(tria_dummy, mpi_comm);

    tria.create_triangulation(description);
  }

  pcout << "\tNumber of active cells:       " << tria.n_global_active_cells()
        << std::endl;

  setup();
  pcout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  u_old = -84e-3;
  u     = u_old;

  output_results();

  assemble_time_independent_matrix();

  // M/dt + A
  system_matrix.copy_from(mass_matrix_dt);
  system_matrix.add(+1, laplace_matrix);

  amg_preconditioner.initialize(system_matrix);

  while (time <= time_end)
    {
      time += dt;
      Iapp->set_time(time);

      ionic_model->solve(u_old);
      assemble_time_terms();

      solve();
      pcout << "Solved at t = " << time << std::endl;
      ++time_step;

      if ((time_step % 10 == 0))
        output_results();

      u_old = u;
    }
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Monodomain::Parameters     monodomain_params;
  BuenoOrovio::Parameters    ionic_model_params;
  Monodomain::Parameters     monodomain_params;
  BuenoOrovio::Parameters    ionic_model_params;
  AppliedCurrent::Parameters applied_current_params;
  ParameterAcceptor::initialize("../parameters.prm");

  Monodomain problem(monodomain_params, ionic_model_params, applied_current_params);
  Monodomain problem(monodomain_params, ionic_model_params, applied_current_params);

  problem.run();

  return 0;
}


