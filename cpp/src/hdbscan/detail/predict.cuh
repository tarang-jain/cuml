/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/linalg/unary_op.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuml/neighbors/knn.hpp>

#include <thrust/transform.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

/**
 * Predict the cluster label and the probability of the label for new points.
 * The returned labels are those of the original clustering found by ``clusterer``,
 * and therefore are not (necessarily) the cluster labels that would
 * be found by clustering the original data combined with
 * the prediction points, hence the 'approximate' label.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data the PredictionData object created during fit
 * @param[in] X input data points (size m * n)
 * @param[in] prediction_points input prediction points (size n_prediction_points * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] n_prediction_points number of prediction points
 * @param[in] metric distance metric to use
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[out] out_labels output cluster labels
 * @param[out] out_probabilities output probabilities
 */
template <typename value_idx, typename value_t>
void _approximate_predict(const raft::handle_t& handle,
                          Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                          Common::PredictionData<value_idx, value_t>& prediction_data,
                          const value_t* X,
                          const value_t* prediction_points,
                          size_t m,
                          size_t n,
                          size_t n_prediction_points,
                          raft::distance::DistanceType metric,
                          int min_samples,
                          value_t* input_core_dists,
                          value_t* out_labels,
                          value_t* out_probabilities)
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_leaves   = condensed_tree.get_n_leaves();

  std::vector<value_t*> inputs;
  inputs.push_back(const_cast<value_t*>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<int64_t> int64_indices(min_samples * n_prediction_points * 2, stream);
  rmm::device_uvector<value_idx> indices(min_samples * n_prediction_points * 2, stream);
  rmm::device_uvector<value_t> dists(min_samples * n_prediction_points * 2, stream);
  rmm::device_uvector<value_t> prediction_core_dists(n_prediction_points, stream);

  // perform knn
  brute_force_knn(handle,
    X,
    sizes,
    n,
    const_cast<value_t*>(prediction_points),
    n_prediction_points,
    int64_indices.data(),
    dists.data(),
    min_samples * 2,
    true,
    true,
    metric);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  // Slice core distances (distances to kth nearest neighbor)
  thrust::transform(exec_policy, indices, indices + n_prediction_points, prediction_core_dists.data(), [dists = dists.data()] __device__(value_idx row) {
    return dists[row * min_samples * 2 + (min_samples - 1)];
  });

  // convert from current knn's 64-bit to 32-bit.
  thrust::transform(exec_policy,
                    int64_indices.data(),
                    int64_indices.data() + int64_indices.size(),
                    indices.data(),
                    [] __device__(int64_t in) -> value_idx { return in; });

  rmm::device_uvector<value_t> min_mr_dists(n_prediction_points, stream);
  rmm::device_uvector<value_t> min_mr_indices(n_prediction_points, stream);
  
  int n_blocks = raft::ceildiv(n_prediction_points, tpb);
  // get nearest neighbors for each prediction point in mutual reachability space
  min_mutual_reachability_kernel<<<n_blocks, tpb, 0, stream>>>(input_core_dists,
                                 prediction_core_dists.data(),
                                 dists.data(),
                                 indices.data(),
                                 n_prediction_points,
                                 min_samples,
                                 min_mr_dists.data(),
                                 min_mr_indices.data());
  
  rmm::device_uvector<value_t> prediction_lambdas(n_prediction_points, stream);

  // obtain lambda values from minimum mutual reachability distances.
  thrust::transform(exec_policy,
    min_mr_dists.data(),
    min_mr_dists.data() + n_prediction_points,
    prediction_lambdas.data(),
    [] __device__(valuue_t dist) {
      if (dist > 0) return (1 / dist); 
      return std::numeric_limits<value_t>::max();});
  
  rmm::device_uvector<value_idx> index_into_children(n_edges + 1, stream);
  auto counting = thrust::make_counting_iterator<value_idx>(0);

  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(children, counting)),
    thrust::make_zip_iterator(thrust::make_tuple(children + n_edges, counting + n_edges)),
    index_op);
  cluster_probability_kernel(min_mr_indices.data(),
        prediction_lambdas.data(),
        index_into_children.data(),
        prediction_data.get_cluster_map(),
        prediction_data.get_deaths(),
        parents,
        n_leaves,
        n_prediction_points,
        out_labels,
        out_probabilities);
  
}

};  // end namespace Predict
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
