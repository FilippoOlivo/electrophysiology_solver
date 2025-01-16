#include <math.h>
#include <algorithm>
#include <iostream>
#include <set>
#include <utility>
#include <vector>


class KNN
{
public:
  KNN(int k)
    : k(k)
  {}

  template <int dim>
  std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
  compute_knn(std::vector<std::vector<double>> &points)
  {
    int n = points.size();

    std::vector<std::vector<double>> distance_matrix(n,
                                                     std::vector<double>(n, 0));
    compute_distance_matrix<3>(points, distance_matrix);


    std::vector<std::vector<int>> edge_index(
      n * k, std::vector<int>(2, 0));
    std::vector<std::vector<double>> edge_distances(n * k, std::vector<double>(6, 0));
    for (int i = 0; i < n; i++)
      {
        std::vector<int> sorted_idx(argsort(distance_matrix[i]));
        for (int j = 0; j < k; j++)
          {
            edge_index[i * k + j]     = {sorted_idx[j + 1], i};
            edge_distances[i * k + j][3] = points[i][0];
            edge_distances[i * k + j][4] = points[i][1];
            edge_distances[i * k + j][5] = points[i][2];

            edge_distances[i * k + j][0] = points[sorted_idx[j + 1]][0];
            edge_distances[i * k + j][1] = points[sorted_idx[j + 1]][1];
            edge_distances[i * k + j][2] = points[sorted_idx[j + 1]][2];
          }
      }
    return std::make_pair(edge_index, edge_distances);
  }

private:
  int k;

  template <int dim>
  void
  compute_distance_matrix(std::vector<std::vector<double>>          &points,
                          std::vector<std::vector<double>> &distance_matrix)
  {
    int n = points.size();
    for (int i = 0; i < n; i++)
      {
        for (int j = i + 1; j < n; j++)
          {
            double distance       = compute_distance<dim>(points[i], points[j]);
            distance_matrix[i][j] = distance;
            distance_matrix[j][i] = distance;
          }
      }
  }

  template <int dim>
  double
  compute_distance(std::vector<double> i, std::vector<double> j)
  {
    double distance = 0;
    for (int k = 0; k < dim; k++)
      distance += std::pow((i[k] - j[k]), 2);
    return std::sqrt(distance);
  }

  template <typename T>
  std::vector<int>
  argsort(std::vector<T> &values)
  {

    std::vector<int> indices(values.size());
    int size = indices.size();
    for (int i = 0; i < size; ++i)
      {
        indices[i] = i;
      }

    std::partial_sort(indices.begin(),
                      indices.begin() + k + 1,
                      indices.end(),
                      [&values](int i1, int i2) {
                        return values[i1] < values[i2];
                      });
    return std::vector<int>(indices.begin(), indices.begin() + k + 1);
  }
};