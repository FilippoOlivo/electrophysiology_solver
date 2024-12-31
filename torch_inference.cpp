#include <torch/torch.h>
#include <torch/script.h>
#include "torch_inference.hpp"
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

TorchInference::TorchInference(std::string filename,
               std::vector<std::vector<unsigned int>> edge_index,
               std::vector<std::vector<double>> edge_attr)
  :  edge_index(to_tensor(edge_index))
  ,  edge_attr(to_tensor(edge_attr))
  ,  model(std::make_shared<torch::jit::Module>(torch::jit::load(filename)))
  {};

template<typename T>
torch::Tensor TorchInference::to_tensor(std::vector<std::vector<T>> vector)
{
    auto type = get_tensor_type<T>();

    long int size = vector.size();
    long int n_dims = vector[0].size();
    torch::Tensor tensor = torch::ones({size, n_dims}, type);

    for(long int i = 0; i < size; i++)
      {
        for(long int j = 0; j < n_dims; j++)
          tensor[i][j] = vector[i][j];
      }
    return tensor;
};

torch::Tensor to_tensor(
    std::array<LinearAlgebra::distributed::Vector<double>, 3> &vector,
    IndexSet &locally_owned_dofs
    )
  {
    long int size = vector[0].size();
    long int n_dims = vector.size();
    torch::Tensor tensor = torch::ones({size, n_dims});
    long int i = 0;
    for (auto idx : locally_owned_dofs)
      {
        for (long int j = 0; j < n_dims; j++)
          tensor[i][j] = vector[j][idx];
        i++;
      }
    return tensor;
  };

  template<typename T>
  auto TorchInference::get_tensor_type()
    {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
          return torch::TensorOptions().dtype(torch::kFloat32);
      } else if constexpr (std::is_same_v<T, unsigned int>) {
          return torch::TensorOptions().dtype(torch::kInt64);
      }
    }