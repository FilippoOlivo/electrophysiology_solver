#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <cnpy.h>
#include <mpi.h>

#include <iostream>
#include <set>
#include <utility>

#include "knn.hpp"

class GraphSaver
{
public:
  GraphSaver(unsigned int mpi_rank, unsigned int mpi_size, MPI_Comm &mpi_comm)
    : mpi_rank(mpi_rank)
    , mpi_size(mpi_size)
    , mpi_comm(mpi_comm)
    , local_size(0)
  {}

  template <int dim>
  void
  build_graph(DoFHandler<dim> &dof_handler, MappingQ<dim> mapping)
  {
    auto [global_to_local_map, local_points] =
      extract_local_points(dof_handler, mapping);
    save_points<double, dim>(local_points, "points");

    KNN knn(6);
    auto [edge_index, edge_attr] = knn.compute_knn(local_points);
    save_points<unsigned int, 2>(edge_index, "edges");
    save_attr<double>(edge_attr);
  }
  void
  save_snapshot(IndexSet locally_owned_dofs,
                std::array<LinearAlgebra::distributed::Vector<double>, 3> vec,
                double                                                    time)
  {
    std::vector<std::vector<double>> snapshot_vector(local_size,
                                                     std::vector<double>(3));
    unsigned int                     i = 0;
    for (auto idx : locally_owned_dofs)
      {
        std::vector<double> temp(3);
        for (unsigned int j = 0; j < 3; j++)
          snapshot_vector[i][j] = vec[j][idx];
        i++;
      }
    char buffer[30];
    snprintf(buffer, sizeof(buffer), "snapshot/%.4f_values", time);
    std::string filename(buffer);
    save_points<double, 3>(snapshot_vector, filename);
  }

private:
  unsigned int          mpi_rank;
  unsigned int          mpi_size;
  MPI_Comm              mpi_comm;
  std::vector<Point<3>> nodes;
  unsigned int          local_size;

  template <int dim>
  std::pair<std::map<types::global_dof_index, unsigned int>,
            std::vector<Point<dim>>>
  extract_local_points(DoFHandler<dim> &dof_handler, MappingQ<dim> &mapping)
  {
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    std::map<types::global_dof_index, Point<3>> points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);

    std::map<types::global_dof_index, unsigned int> global_to_local_map;
    std::vector<Point<dim>>                         local_points;
    unsigned int                                    i = 0;
    local_size = dof_handler.n_locally_owned_dofs();
    local_points.resize(local_size);
    for (auto idx : locally_owned_dofs)
      {
        local_points[i]          = points[idx];
        global_to_local_map[idx] = i;
        ++i;
      }
    return std::make_pair(global_to_local_map, local_points);
  }

  template <typename T, int dim, typename S>
  void
  save_points(std::vector<S> &points, std::string base_filename)
  {
    std::vector<T> points_flatten(points.size() * dim);
    for (unsigned int i = 0; i < points.size(); ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          points_flatten[i * dim + j] = points[i][j];
      }
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "_%.2d.npy", mpi_rank);
    std::string end_filename(buffer);
    std::string filename = base_filename + end_filename;
    cnpy::npy_save(filename, points_flatten.data(), {points.size(), dim}, "w");
  }

  template <typename T>
  void
  save_attr(std::vector<T> &vec)
  {
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "attr_%.2d.npy", mpi_rank);
    std::string filename(buffer);
    //std::string filename = base_filename + end_filename;
    cnpy::npy_save(filename, vec.data(), {vec.size()}, "w");
  }
};
