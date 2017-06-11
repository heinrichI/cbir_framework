# Search
4 types of search are supported:
* exhaustive
* sdc exhaustive
* adc exhaustive
* inverted multi-index

#### Exhaustive search
requires:
- [global descriptors](#global_descriptors)

#### SDC exhaustive search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)
- [centroids pairwise distances](#centroids_pairwise_distances)

#### ADC exhaustive search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)

#### Inverted multi-index search
requires:
- [centroids](#centroids)
- [pq codes](#pq_codes)


# Descriptors computation
* <a name="global_descriptors">global descriptors</a>
    * compute from [image](examples/different_descriptors.ipynb)
    * compute from local descriptors
* local descriptors
    * compute from image

# Quantization
* finding <a name="centroids">centroids</a>
* quantizing global descriptors to <a name="pq codes">pq codes</a>
* finding <a name="centroids_pairwise_distances">centroids pairwise distances</a>

# Sampling
It`s often enough to quantize only sample from descriptors.
Example.

# Evaluation

# Plotting
