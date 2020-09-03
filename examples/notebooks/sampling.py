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
from core.common.path_helper import DataStoreHelper
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Sampling

# %%
ds_helper=DataStoreHelper(r'C:\data\computation\brodatz')


# %%
sifts_ds = ds_helper.local_descriptors_ds('sifts', False)
sample_part=0.3
sifts_sample_ds =  ds_helper.local_descriptors_sample_ds('sifts', sample_part)
steps.sampling_step(sifts_ds, sample_part, sifts_sample_ds)


# %%
print_ds_items_info(sifts_ds)
print_ds_items_info(sifts_sample_ds)


# %%



