class SearchConfig:
    def __init__(self, transformers, searcher, base_native_ids_ds):
        self.transformers = transformers
        self.searcher = searcher
        self.base_native_ids_ds = base_native_ids_ds

"""
def build_object_from_json_dict(obj):
    type_ = obj['type']
    params_ = obj['params']
    obj = create_single_by_type_name_and_kwargs(type_, params_)
    return obj


def build_data_stores_dict_from_json_dict(obj):
    data_stores_dict = {}
    data_store_json_arr = obj['data_stores']
    for ds_json in data_store_json_arr:
        ds_name = ds_json['name']
        ds = build_object_from_json_dict(ds_json)
        data_stores_dict[ds_name] = ds
    return data_stores_dict


def build_pq_quantizers_dict_from_json_dict(obj):
    pq_quantizers_dict = {}
    pq_quantizer_json_arr = obj['pq_quantizers']
    for pq_quantizer_json in pq_quantizer_json_arr:
        pq_quantizer_name = pq_quantizer_json['name']
        pq_quantizer = build_object_from_json_dict(pq_quantizer_json)
        pq_quantizers_dict[pq_quantizer_name] = pq_quantizer
    return pq_quantizers_dict

def build_transformers_array_from_json_dict(obj, pq_quantizers_dict):
    transformers_arr=[]
    json_arr = obj['transformers']
    for json_ in json_arr:
        # name = json_['name']
        type_=json['type']
        params=json['params']
        for p in params:
            if p=='pq_quantizer':
                pq_quantizer_name=pq_quantizers_dict[p]
                pq_quantizer=pq_quantizers_dict[pq_quantizer_name]
                params[p]=pq_quantizer
        pq_quantizer = build_object_from_json_dict(pq_quantizer_json)
        pq_quantizers_dict[pq_quantizer_name] = pq_quantizer
    return pq_quantizers_dict


config_dict = {
    'transformers': [
        {'type': 'BytesToNdarray',
         'kwargs': {}
         },
        {'type': 'NdarrayToOpencvMatrix',
         'kwargs': {}
         },
        {'type': 'OpencvMatrixToHistogram',
         'kwargs': {}
         },
    ],
    'base_image_native_ids': {
        {'type': 'BytesToNdarray',
         'kwargs': {}
         }
    },
    'searcher': {
        'type': 'ExhaustiveSearcher',
        'kwargs': {}
    }
}
"""