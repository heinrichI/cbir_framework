# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __init__ import *
import numpy as np
import core.steps as steps
from core import data_store as ds
from core.common.ds_utils import print_ds_items_info
from core.evaluation.ground_truth import BrodatzGroundTruth
from core.evaluation.retrieval_perfomance import PrecisionRecallAveragePrecisionEvaluator
from core.common.path_helper import DataStoreHelper
import os
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Evaluate search
# Compute search precision, recall, mAP perfomance. 
# 
# PrecisionRecallAveragePrecisionEvaluator builds perfomance_arr of shape(4, n_nearest), where:
# - perfomance_arr[0,:] - n_nearest cutoffs
# - perfomance_arr[1,:] - precisions
# - perfomance_arr[2,:] - recalls
# - perfomance_arr[3,:] - mAPs
# 
# Here we will save such arrays in csv files (one perfomance_arr - one file)

# %%
ds_helper=DataStoreHelper(r'C:\data\computation\brodatz')


# %%
def evaluate_search(search_type, descriptor_name, pq_params):
    if search_type=='adc' or search_type=='sdc' or search_type=='imi':
        neighbors_ids_ds=ds_helper.pq_search_neighbors_ids_ds(search_type, descriptor_name, pq_params)
        search_perfomances_ds=ds_helper.pq_search_perfomances_ds(search_type, descriptor_name, pq_params)
    else:
        neighbors_ids_ds=ds_helper.ex_search_neighbors_ids_ds(descriptor_name)
        search_perfomances_ds=ds_helper.ex_search_perfomances_ds(descriptor_name)
        
    ground_truth = BrodatzGroundTruth()
    evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)

    steps.evaluation_step(neighbors_ids_ds, evaluator, search_perfomances_ds)


# %%
K_arr = [2 ** i for i in [4,5,6,7,8]]
m_arr = [m for m in [1, 2, 4, 8, 16]]
pq_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr]

bytes_free=1 << 34
imi_pq_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr if 4 * K ** m < bytes_free]
imi_pq_params_arr.remove({'n_clusters': 128, 'n_quantizers': 4})


# %%
descriptor_names=['histograms', 'lbphistograms', 'glcms']
choosen_bovwproductbincounts_pq_params= [{'n_clusters': K, 'n_quantizers': m} for K,m in [(64,4),(128,1),(16,1)]]
descriptor_names+=ds_helper.bovw_descriptors_names('bovwproductbincounts',choosen_bovwproductbincounts_pq_params)


# %%
for descriptor_name in descriptor_names:
    evaluate_search('ex', descriptor_name, None)


# %%
for descriptor_name in descriptor_names:
    for pq_params in pq_params_arr:
        evaluate_search('adc', descriptor_name, pq_params)


# %%
for descriptor_name in descriptor_names:
    for pq_params in pq_params_arr:
        evaluate_search('sdc', descriptor_name, pq_params)


# %%
for descriptor_name in descriptor_names:
    for pq_params in imi_pq_params_arr:
        evaluate_search('imi', descriptor_name, pq_params)


# %%



# %%



