# Search
4 types of search are supported:
* exhaustive
* sdc exhaustive
* adc exhaustive
* inverted multi-index

### Exhaustive search
requires:
- [global descriptors](#global_descriptors)

### SDC exhaustive search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)
- [centroids pairwise distances](#centroids_pairwise_distances)

### ADC exhaustive search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)

### Inverted multi-index search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)


# Descriptors computation
* <a name="global_descriptors">global descriptors</a>
    * [compute from image](/examples/notebooks/descriptors_computation/compute_global_descriptors_from_image.ipynb)
    * [compute from local descriptors](/examples/notebooks/descriptors_computation/compute_global_descriptors_from_local_descriptors.ipynb)
* local descriptors
    * [compute from image](/examples/notebooks/descriptors_computation/compute_local_descriptors_from_image.ipynb)

# Quantization
* finding <a name="centroids">[centroids](/examples/notebooks/quantization/finding_centroids.ipynb)</a>
* quantizing global descriptors to <a name="pq_codes">[pq codes](/examples/notebooks/quantization/quantize_global_descriptors_to_pqcodes.ipynb)</a>
* finding <a name="centroids_pairwise_distances">[centroids pairwise distances](/examples/notebooks/quantization/compute_centroids_pairwise_distances.ipynb)</a>

# Sampling
It`s often enough to quantize only sample from descriptors.

[Example(sampling sifts)](/examples/notebooks/sampling.ipynb).

# Evaluation
Step to evaluate search perfomance.

[Example](/examples/notebooks/evaluate_search.ipynb).

# Plotting
* compare descriptors for exhaustive search [Example->](/examples/notebooks/plotting/plot_exhaustive_search_perfomance_n_nearest.ipynb)
* compare memory for descriptors for exhaustive search [Example->](/examples/notebooks/plotting/plot_exhaustive_search_perfomance_memory.ipynb)
* compare quantization parameters for pq search techniques(adc, sdc, imi) [Example->](/examples/notebooks/plotting/plot_search_perfomance_pq_params.ipynb)
* compare pq search types(adc, sdc, imi) [Example->](/examples/notebooks/plotting/plot_search_perfomance_search_types.ipynb)
