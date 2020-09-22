# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __init__ import *
import core.steps as steps
from core import data_store as ds
from core import transformer as tr
from core.common.ds_utils import print_ds_items_info
from core.search.inverted_multi_index_searcher import InvertedMultiIndexSearcher
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters, build_pq_params_str
from core.common.path_helper import DataStoreHelper
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Inverted multi-index search

# %%
ds_helper=DataStoreHelper(r'C:\data\computation\brodatz')


# %%
def imi_search(descriptors_name, pq_params):
    centroids_ds=ds_helper.centroids_ds(descriptor_name, pq_params)
    centroids = ds.get_as_array(centroids_ds)
  
    pqcodes_ds = ds_helper.pqcodes_ds(descriptor_name, pq_params)
    ids, pqcodes = ds.get_as_array(pqcodes_ds, return_ids=True)
    
    searcher_ = InvertedMultiIndexSearcher(ids, centroids, x_pqcodes=pqcodes)
    n_nearest = 25
    
    neighbors_ids_ds=ds_helper.pq_search_neighbors_ids_ds('imi', descriptor_name, pq_params)
    query_descriptors_ds=ds_helper.global_descriptors_ds(descriptor_name)
    
    steps.search_step(query_descriptors_ds, searcher_, n_nearest, neighbors_ids_ds)


# %%
descriptor_names=['histograms', 'lbphistograms', 'glcms']
choosen_bovwproductbincounts_pq_params= [{'n_clusters': K, 'n_quantizers': m} for K,m in [(64,4),(128,1),(16,1)]]
descriptor_names+=ds_helper.bovw_descriptors_names('bovwproductbincounts',choosen_bovwproductbincounts_pq_params)


# %%
K_arr = [2 ** i for i in [4,5,6,7,8]]
m_arr = [m for m in [1, 2, 4, 8, 16]]
# inverted multi-index will take 4 * K ** m bytes of memory, so we need to put restrictions on pq_params
bytes_free=1 << 34
imi_pq_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr if 4 * K ** m < bytes_free]
imi_pq_params_arr.remove({'n_clusters': 128, 'n_quantizers': 4})
# current implementation works too slow with (m>2, K>32), e.g. (2,256) - fast, but (4,64) - very slow.
print(imi_pq_params_arr)


# %%
for descriptor_name in descriptor_names:
    for pq_params in imi_pq_params_arr:
        imi_search(descriptor_name, pq_params)


# %%



