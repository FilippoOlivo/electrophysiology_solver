#include "utils.hpp"
#include <deal.II/lac/la_parallel_vector.h>
#include <typeinfo>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <cmath>
#include <fstream>


inline std::vector<double> gather_vector(const unsigned int mpi_rank, const unsigned int mpi_size,
              LinearAlgebra::distributed::Vector<double> vector, MPI_Comm mpi_comm)
{
  std::vector<double> local_elements(vector.locally_owned_size(), 1000);
  int local_size = vector.locally_owned_size();

  unsigned int i = 0;
  for (const auto& idx : vector.locally_owned_elements()) {
      local_elements[i] = vector[idx];
      ++i;
  }
  std::vector<double> entire_elements;

  std::vector<int> per_proc_size(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT, per_proc_size.data(), 1,
                MPI_INT, mpi_comm);

  std::vector<int> displacement(mpi_size, 0);
  for (unsigned int i = 1; i < mpi_size; i++)
    {
      displacement[i] = displacement[i - 1] + per_proc_size[i - 1];
    }


  if (mpi_rank==0)
    entire_elements.resize(vector.size());

  MPI_Gatherv(local_elements.data(), // send buffer
           local_size, // send size
           MPI_DOUBLE, // type
           mpi_rank == 0 ? entire_elements.data() : nullptr, // receive buffer
           per_proc_size.data(), // receive size
           displacement.data(),
           MPI_DOUBLE, // receive type
           0, // main processor
           mpi_comm // MPI Comminicator
           );
  MPI_Barrier(mpi_comm);
  return entire_elements;
};

template <typename T>
inline void save_vector_as_binary(const std::vector<T> &vec, const std::string &filename,
                      unsigned int vec_size)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      throw std::runtime_error("Error opening file for writing.");
    }

  file.write(reinterpret_cast<const char *>(&vec_size), sizeof(vec_size));

  file.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(T));
  file.close();
};


inline void save_snapshot(const unsigned int mpi_rank, const unsigned int mpi_size,
              std::array<LinearAlgebra::distributed::Vector<double>, 3> vec,
              double time, MPI_Comm mpi_comm)
  {
    std::vector<double> global_vector;
    if (mpi_rank==0)
        global_vector.resize(vec[0].size() * vec.size(), 10);
    for (unsigned int i=0; i<vec.size(); i++)
      {
        auto val = gather_vector(mpi_rank, mpi_size, vec[i], mpi_comm);
        if (mpi_rank==0){
          unsigned int offset = vec[0].size() * i;
          for (unsigned int j = 0; j<val.size(); j++){
            global_vector[j+offset] = val[j];
          }
        }
        MPI_Barrier(mpi_comm);
      }

    if (mpi_rank == 0){
        char buffer_w[100];
        snprintf(buffer_w, sizeof(buffer_w), "snapshot/%.3f_w.bin", time);
        std::string filename_w(buffer_w);
        save_vector_as_binary(global_vector, filename_w, vec[0].size());
      }

    MPI_Barrier(mpi_comm);

  }