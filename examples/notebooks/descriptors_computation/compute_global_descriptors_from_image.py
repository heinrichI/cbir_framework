# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __init__ import *
import numpy as np
import core.steps as steps
from core import data_store as ds
from core import transformer as tr
from core.common.ds_utils import print_ds_items_info
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters, build_pq_params_str
from core.common.path_helper import DataStoreHelper
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# %%
ds_helper=DataStoreHelper(r'C:\data\computation\brodatz')

# %% [markdown]
# # Compute global descriptors from local descriptors
# (siftset->bovwsiftbincount)

# %%
sifts_ds=ds_helper.local_descriptors_ds('sifts')
with sifts_ds:
    sifts_list_ds = ds.ListDatastore(sifts_ds.get_items_sorted_by_ids())


# %%
def compute_global_descriptor_from_sifts(global_descriptor_name, pq_params):
    local_descriptors_ds=sifts_list_ds
    
    centroids_ds =ds_helper.centroids_ds('sifts', pq_params)
    centroids = ds.get_as_array(centroids_ds)
    pq_quantizer = restore_from_clusters(centroids)

    if global_descriptor_name=='bovwproductbincounts':
        transformers=[tr.ArraysToProductBinCount(pq_quantizer)]
    elif global_descriptor_name=='bovwbincounts':
        if pq_quantizer.max_scalar_index>2**19:
            print(pq_params, ": too much memory for descriptors")
            return
        transformers=[tr.ArraysToBinCount(pq_quantizer)]

    global_descriptors_ds=ds_helper.bovw_descriptors_ds(global_descriptor_name, pq_params)

    steps.transform_step(local_descriptors_ds, transformers, global_descriptors_ds)


# %%
K_arr = [2 ** i for i in [4,5,6,7,8]]
m_arr = [m for m in [1, 2, 4, 8, 16]]
pq_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr]


# %%
for pq_params in pq_params_arr:
    compute_global_descriptor_from_sifts('bovwbincounts',pq_params)


# %%
for pq_params in pq_params_arr:
    compute_global_descriptor_from_sifts('bovwproductbincounts',pq_params)


# %%



