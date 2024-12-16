#include "utils.hpp"
#include <deal.II/lac/la_parallel_vector.h>
#include <typeinfo>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <cmath>
#include <fstream>
#include <cnpy.h>

template <typename T>
inline std::vector<T> gather_vector(const int mpi_rank,
              const int mpi_size,
              std::vector<T> vector,
              int total_size,
              MPI_Comm mpi_comm)
{
  MPI_Datatype type;
  if constexpr (std::is_same<T, int>::value)
    type = MPI_INT;
  else if constexpr (std::is_same<T, double>::value)
    type = MPI_DOUBLE;
  else
    throw std::runtime_error("Type not supported");

  std::vector<T> entire_elements;
  std::vector<int> sizes(mpi_size);
  int local_size = vector.size();
  MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1,
                MPI_INT, mpi_comm);
  std::vector<int> displacement(mpi_size, 0);
  for (int i = 1; i < mpi_size; i++)
  {
    displacement[i] = displacement[i - 1] + sizes[i - 1];
  }
  if (mpi_rank==0)
    entire_elements.resize(total_size);
  MPI_Gatherv(vector.data(),
           local_size,
           type,
           mpi_rank == 0 ? entire_elements.data() : nullptr,
           sizes.data(),
           displacement.data(),
           type,
           0,
           mpi_comm
           );
  return entire_elements;
};

template <typename T>
inline void save_vector(const std::vector<T> &vec, const std::string &filename)
{
  cnpy::npy_save(filename, vec.data(), {vec.size()}, "w");
};

inline void save_snapshot(const unsigned int mpi_rank,
              const unsigned int mpi_size,
              std::array<LinearAlgebra::distributed::Vector<double>, 3> vec,
              double time,
              IndexSet locally_owned_dofs,
              MPI_Comm mpi_comm)
  {
    std::vector<double> global_vector;
    if (mpi_rank==0)
      {
        global_vector.resize(vec[0].size() * vec.size());
      }
    for (unsigned int i=0; i<vec.size(); i++)
      {

        std::vector<double> local_elements(vec[i].locally_owned_size());
        unsigned int j = 0;
        for (const auto& idx : locally_owned_dofs) {
            local_elements[j] = vec[i][idx];
            ++j;
        }
        auto val = gather_vector<double>(mpi_rank, mpi_size, local_elements,
                                         vec[i].size(), mpi_comm);
        if (mpi_rank==0){
          unsigned int offset = vec[0].size() * i;
          for (unsigned int j = 0; j<val.size(); j++){
            global_vector[j+offset] = val[j];
          }
        }
        MPI_Barrier(mpi_comm);
      }
    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0){
        char buffer_w[100];
        snprintf(buffer_w, sizeof(buffer_w), "snapshot/%.4f.npy", time);
        std::string filename_w(buffer_w);
        save_vector(global_vector, filename_w);
      }
    MPI_Barrier(mpi_comm);
  }

template <int dim>
void
save_dofs_location(DoFHandler<dim> &dof_handler, IndexSet &locally_owned_dofs,
                   MappingQ<dim> &mapping, int mpi_rank, int mpi_size,
                   MPI_Comm mpi_comm)
{
  std::map<types::global_dof_index, Point<3>> locations =
    DoFTools::map_dofs_to_support_points(mapping, dof_handler);
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

  for (unsigned int i=0; i<dof_handler.n_locally_owned_dofs(); i++)
    {
      x[i] = local_locations[i][0];
      y[i] = local_locations[i][1];
      z[i] = local_locations[i][2];
    }

  /*
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
  */

  auto global_x = gather_vector(mpi_rank, mpi_size, x, dof_handler.n_dofs(), mpi_comm);
  auto global_y = gather_vector(mpi_rank, mpi_size, y, dof_handler.n_dofs(), mpi_comm);
  auto global_z = gather_vector(mpi_rank, mpi_size, z, dof_handler.n_dofs(), mpi_comm);

  if (mpi_rank == 0)
    {
    save_vector(global_x, "global_x.npy");
    save_vector(global_y, "global_y.npy");
    save_vector(global_z, "global_z.npy");
    }
  MPI_Barrier(mpi_comm);
}

