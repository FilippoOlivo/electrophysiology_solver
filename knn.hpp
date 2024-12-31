#include <math.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <utility>
#include <vector>


class KNN
{
public:
  KNN(unsigned int k)
    : k(k)
  {}

  template <int dim>
  std::pair<std::vector<std::vector<unsigned int>>, std::vector<double>>
  compute_knn(std::vector<Point<dim>> &points)
  {
    unsigned int n = points.size();

    std::vector<std::vector<double>> distance_matrix(n,
                                                     std::vector<double>(n, 0));
    compute_distance_matrix<3>(points, distance_matrix);


    std::vector<std::vector<unsigned int>> edge_index(
      n * k, std::vector<unsigned int>(2, 0));
    std::vector<double> edge_distances(n * k, 0);
    for (unsigned int i = 0; i < n; i++)
      {
        std::vector<unsigned int> sorted_idx(argsort(distance_matrix[i]));
        for (unsigned int j = 0; j < k; j++)
          {
            edge_index[i * k + j]     = {sorted_idx[j + 1], i};
            edge_distances[i * k + j] = distance_matrix[i][j + 1];
          }
      }
    return std::make_pair(edge_index, edge_distances);
  }

private:
  unsigned int k;

  template <int dim>
  void
  compute_distance_matrix(std::vector<Point<dim>>          &points,
                          std::vector<std::vector<double>> &distance_matrix)
  {
    for (unsigned int i = 0; i < points.size(); i++)
      {
        for (unsigned int j = i + 1; j < points.size(); j++)
          {
            double distance       = compute_distance(points[i], points[j]);
            distance_matrix[i][j] = distance;
            distance_matrix[j][i] = distance;
          }
      }
  }

  template <int dim>
  double
  compute_distance(Point<dim> i, Point<dim> j)
  {
    double distance = 0;
    for (unsigned int k = 0; k < dim; k++)
      distance += std::pow((i[k] - j[k]), 2);
    return std::sqrt(distance);
  }

  template <typename T>
  std::vector<unsigned int>
  argsort(std::vector<T> &values)
  {
    // Create a vector of indices [0, 1, 2, ..., n-1]
    std::vector<unsigned int> indices(values.size());
    for (unsigned int i = 0; i < indices.size(); ++i)
      {
        indices[i] = i;
      }

    std::partial_sort(indices.begin(),
                      indices.begin() + k + 1,
                      indices.end(),
                      [&values](unsigned int i1, unsigned int i2) {
                        return values[i1] < values[i2];
                      });
    return std::vector<unsigned int>(indices.begin(), indices.begin() + k + 1);

    // Return only the top k+1 indices
    return std::vector<unsigned int>(indices.begin(), indices.begin() + k);
  }
};