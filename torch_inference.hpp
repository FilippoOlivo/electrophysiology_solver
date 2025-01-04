#pragma once
#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <iostream>
#include <vector>
#undef Assert
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace dealii;

class TorchInference
{
public:
  TorchInference() = default;
  TorchInference(std::string                            filename,
                 std::vector<std::vector<unsigned int>> edge_index,
                 std::vector<std::vector<double>>       edge_attr);

  template <typename T>
  torch::Tensor
  to_tensor(std::vector<std::vector<T>> vector);

  template <typename T>
  torch::Tensor
  to_tensor(std::vector<T> vector);

  torch::Tensor
  to_tensor(
    std::array<dealii::LinearAlgebra::distributed::Vector<double>, 3> &vector,
    IndexSet &locally_owned_dofs);

  torch::Tensor
  run(torch::Tensor x);

private:
  torch::Tensor                       edge_index;
  torch::Tensor                       edge_attr;
  std::shared_ptr<torch::jit::Module> model;

  template <typename T>
  auto
  get_tensor_type();
};
