#include "utils.hpp"
#include <deal.II/lac/la_parallel_vector.h>
#include <typeinfo>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <cmath>
#include <fstream>

#include <cnpy.h>
#pragma once 
class GatherTool {
public:
  unsigned int mpi_rank;
  MPI_Comm mpi_comm;
  GatherTool() = default;

  template <int dim>
  void setup(unsigned int mpi_rank, unsigned int mpi_size, DoFHandler<dim> &dof_handler, MPI_Comm mpi_comm){ 
    this->mpi_comm = mpi_comm; this->mpi_size = mpi_size; this->mpi_rank = mpi_rank;
    local_size = dof_handler.n_locally_owned_dofs();
    
    sizes.resize(mpi_size);
    displacement.resize(mpi_size);
    MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, mpi_comm);
    displacement[0] = 0;
    for (unsigned int i = 1; i < mpi_size; ++i) {
        displacement[i] = displacement[i - 1] + sizes[i - 1];
    }
    total_size = dof_handler.n_dofs();
  }


  template <typename T>
  std::vector<T> gather_vector(std::vector<T> vector) {
    
    MPI_Datatype type;
    if constexpr (std::is_same<T, int>::value)
      type = MPI_INT;
    else if constexpr (std::is_same<T, double>::value)
      type = MPI_DOUBLE;
    else
      throw std::runtime_error("Type not supported");

    std::vector<T> entire_elements;
    if (mpi_rank == 0)
      entire_elements.resize(total_size);
    MPI_Gatherv(vector.data(), local_size, type,
                mpi_rank == 0 ? entire_elements.data() : nullptr,
                sizes.data(), displacement.data(), type, 0, mpi_comm);
    MPI_Barrier(mpi_comm);
    return entire_elements;
  }

private:
  unsigned int mpi_size;
  int local_size;
  int total_size;
  std::vector<int> sizes;
  std::vector<int> displacement;
};

template <typename T>
inline void save_vector(const std::vector<T> &vec, const std::string &filename)
{
  cnpy::npy_save(filename, vec.data(), {vec.size()}, "w");
};

inline void save_snapshot(GatherTool &gather_tool,
              IndexSet locally_owned_dofs,
              std::array<LinearAlgebra::distributed::Vector<double>, 3> vec,
              double time)
  {
    std::vector<double> global_vector;
    if (gather_tool.mpi_rank==0)
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
        auto val = gather_tool.gather_vector<double>( local_elements);
        if (gather_tool.mpi_rank==0){
          unsigned int offset = vec[0].size() * i;
          for (unsigned int j = 0; j<val.size(); j++){
            global_vector[j+offset] = val[j];
          }
        }
        MPI_Barrier(gather_tool.mpi_comm);
      }
    if (gather_tool.mpi_rank == 0){
        char buffer_w[100];
        snprintf(buffer_w, sizeof(buffer_w), "snapshot/%.4f.npy", time);
        std::string filename_w(buffer_w);
        save_vector(global_vector, filename_w);
      }
    MPI_Barrier(gather_tool.mpi_comm);
  }

template <int dim>
void
save_dofs_location(DoFHandler<dim> &dof_handler, IndexSet &locally_owned_dofs,
                   MappingQ<dim> &mapping, GatherTool &gather_tool)
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
  auto global_x = gather_tool.gather_vector(x);
  auto global_y = gather_tool.gather_vector(y);
  auto global_z = gather_tool.gather_vector(z);

  if (gather_tool.mpi_rank == 0)
    {
    save_vector(global_x, "global_x.npy");
    save_vector(global_y, "global_y.npy");
    save_vector(global_z, "global_z.npy");
    }
}

