#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#undef Assert
#include <torch/script.h>
#include <torch/torch.h>

#include "torch_inference.hpp"

using namespace dealii;

TorchInference::TorchInference(
  std::string                            filename,
  std::vector<std::vector<int>>          edge_index,
  std::vector<std::vector<double>>       edge_attr,
  int n_elements)
  : edge_index(to_tensor(edge_index).transpose(0, 1).contiguous())
  , edge_attr(to_tensor(edge_attr).contiguous())
  , model(std::make_shared<torch::jit::Module>(torch::jit::load(filename)))
  {  
    inputs.resize(3);
    inputs[1] = this->edge_index;
    inputs[2] = this->edge_attr;
    this->n_elements = n_elements;
  };

template <typename T>
torch::Tensor
TorchInference::to_tensor(std::vector<std::vector<T>> vector)
{
  auto type = get_tensor_type<T>();

  long int      size   = vector.size();
  long int      n_dims = vector[0].size();
  torch::Tensor tensor = torch::ones({size, n_dims}, type);

  for (long int i = 0; i < size; i++)
    {
      for (long int j = 0; j < n_dims; j++)
        tensor[i][j] = vector[i][j];
    }
  return tensor;
};

template <typename T>
torch::Tensor
TorchInference::to_tensor(std::vector<T> vector)
{
  auto type = get_tensor_type<T>();

  long int      size   = vector.size();
  torch::Tensor tensor = torch::ones({size}, type);

  for (long int i = 0; i < size; i++)
    {
      tensor[i] = vector[i];
    }
  return tensor;
};

torch::Tensor
TorchInference::to_tensor(
  std::array<LinearAlgebra::distributed::Vector<double>, 3> &vector,
  IndexSet                                                  &locally_owned_dofs)
{
  long int      n_dims = vector.size();
  torch::Tensor tensor = torch::ones({n_elements, n_dims}, torch::kDouble);
  long int      i      = 0;
  for (auto idx : locally_owned_dofs)
    {
      for (long int j = 0; j < n_dims; j++)
        tensor[i][j] = vector[j][idx];
      i++;
    }
  return tensor;
};

template <typename T>
auto
TorchInference::get_tensor_type()
{
  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
      return torch::TensorOptions().dtype(torch::kDouble);
    }
  else if constexpr (std::is_same_v<T, int>)
    {
      return torch::TensorOptions().dtype(torch::kInt64);
    }
}

void
TorchInference::run(torch::Tensor &res, torch::Tensor &x,  double time)
{ 
  at::Tensor tensor = inputs[3].toTensor();
  for (int i = 0; i < n_elements; ++i) {
    tensor[i][0] = time;
  }
  inputs[0] = x;
  inputs[3] = tensor;
  res = model->forward(inputs).toTensor();
}

void 
TorchInference::run(torch::Tensor &x)
{
  inputs[0] = x;
  x = model->forward(inputs).toTensor();
}

void
TorchInference::run(torch::Tensor &res, torch::Tensor &x)
{ 
  inputs[0] = x;
  res = model->forward(inputs).toTensor();
}