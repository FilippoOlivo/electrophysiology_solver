#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

class TorchInference {
  public:
    TorchInference(std::string filename,
               std::vector<std::vector<unsigned int>> edge_index,
               std::vector<std::vector<double>> edge_attr);

  template<typename T>
  torch::Tensor to_tensor(std::vector<std::vector<T>> vector);


  torch::Tensor to_tensor(
    std::array<dealii::LinearAlgebra::distributed::Vector<double>, 3> &vector,
    dealii::DoFHandler<3> &dof_handler
    );

  void run(torch::Tensor x);

  private:
    torch::Tensor edge_index;
    torch::Tensor edge_attr;
    std::shared_ptr<torch::jit::Module> model;

  template <typename T>
  auto get_tensor_type();
};
