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
  GraphSaver(int mpi_rank, int mpi_size, MPI_Comm &mpi_comm)
    : mpi_rank(mpi_rank)
    , mpi_size(mpi_size)
    , mpi_comm(mpi_comm)
    , local_size(0)
  {}

  template <int dim>
  std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
  build_graph(DoFHandler<dim> &dof_handler, MappingQ<dim> mapping)
  {
    auto local_points = extract_local_points(dof_handler, mapping);
    save_points<double>(local_points, "points");

    KNN knn(6);
    auto [edge_index, edge_attr] = knn.compute_knn<3>(local_points);
    save_points<int>(edge_index, "edges");
    save_points<double>(edge_attr, "attr");
    return std::make_pair(edge_index, edge_attr);
  }
  void
  save_snapshot(IndexSet locally_owned_dofs,
                std::array<LinearAlgebra::distributed::Vector<double>, 3> vec,
                double                                                    time)
  {
    std::vector<std::vector<double>> snapshot_vector(local_size,
                                                     std::vector<double>(3));
    int                              i = 0;
    for (auto idx : locally_owned_dofs)
      {
        std::vector<double> temp(3);
        for (int j = 0; j < 3; j++)
          snapshot_vector[i][j] = vec[j][idx];
        i++;
      }
    char buffer[30];
    snprintf(buffer, sizeof(buffer), "snapshot/%.4f_values", time);
    std::string filename(buffer);

    save_points<double>(snapshot_vector, filename);
  }

  void
  save_snapshot(IndexSet                                   locally_owned_dofs,
                LinearAlgebra::distributed::Vector<double> vec,
                double                                     time,
                std::string                                filename)
  {
    std::vector<double> snapshot_vector(local_size, 0);
    int                 i = 0;
    for (auto idx : locally_owned_dofs)
      {
        snapshot_vector[i] = vec[idx];
        ++i;
      }
    char buffer[30];
    snprintf(buffer, sizeof(buffer), "snapshot/%.4f", time);
    std::string start_filename(buffer);
    filename = start_filename + filename;
    // save_points<double>(snapshot_vector, filename);
    char buffer2[40];
    snprintf(buffer2, sizeof(buffer2), "_%.2d.npy", mpi_rank);
    std::string end_filename(buffer2);
    std::string filename2 = filename + end_filename;
    cnpy::npy_save(filename2,
                   snapshot_vector.data(),
                   {snapshot_vector.size()},
                   "w");
  }

private:
  int                   mpi_rank;
  int                   mpi_size;
  MPI_Comm              mpi_comm;
  std::vector<Point<3>> nodes;
  int                   local_size;

  template <int dim>
  std::vector<std::vector<double>>
  extract_local_points(DoFHandler<dim> &dof_handler, MappingQ<dim> &mapping)
  {
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    std::map<types::global_dof_index, Point<3>> points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);

    std::map<types::global_dof_index, int> global_to_local_map;

    int i      = 0;
    local_size = dof_handler.n_locally_owned_dofs();
    std::vector<std::vector<double>> local_points(local_size,
                                                  std::vector<double>(3));
    for (auto idx : locally_owned_dofs)
      {
        for (int j = 0; j < dim; j++)
          local_points[i][j] = points[idx][j];
        global_to_local_map[idx] = i;
        ++i;
      }
    return local_points;
  }

  template <typename T>
  void
  save_points(std::vector<std::vector<T>> &points, std::string base_filename)
  {
    torch::Tensor tensor;
    if constexpr (std::is_same<T, int>::value)
      {
        tensor = torch::zeros({static_cast<int64_t>(points.size()),
                               static_cast<int64_t>(points[0].size())},
                              torch::kInt);
      }
    else
      {
        tensor = torch::zeros({static_cast<int64_t>(points.size()),
                               static_cast<int64_t>(points[0].size())},
                              torch::kDouble);
      }
    for (unsigned int i = 0; i < points.size(); i++)
      {
        for (unsigned int j = 0; j < points[0].size(); j++)
          tensor[i][j] = points[i][j];
      }
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "_%.2d.pt", mpi_rank);
    std::string end_filename(buffer);
    std::string filename = base_filename + end_filename;
    torch::save(tensor, filename);
  }
};
