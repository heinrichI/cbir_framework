import os
import re


def change_suffix(basedir, extension, new_suffix):
    for fname in os.listdir(basedir):
        if extension in fname:
            result = re.search(r'(pq-[0-9]+-[0-9]+).*', fname)
            print(result.groups())
            pq_params = result.group(1)
            new_name = pq_params + new_suffix + extension
            print(new_name)
            os.rename(os.path.join(basedir, fname), os.path.join(basedir, new_name))

def keep_only_pq_params_str(basedir, extension):
    for fname in os.listdir(basedir):
        if extension in fname:
            result = re.search(r'(pq-[0-9]+-[0-9]+).*', fname)
            print(result.groups())
            pq_params = result.group(1)
            new_name = pq_params
            new_name = new_name + extension
            print(new_name)
            os.rename(os.path.join(basedir, fname), os.path.join(basedir, new_name))

if __name__ == '__main__':
    # basedir = r'C:\Users\Dima\GoogleDisk\notebooks\cbir_framework\examples\sift_1M\ds_data\learn_quantization\sdc-neighbors-ids'
    # basedir = r'C:\Users\Dima\GoogleDisk\notebooks\cbir_framework\examples\sift_1M\ds_data\base_quantization\centroids'
    # change_suffix(basedir, '.sqlite', '_centroids')

    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids\histograms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids\glcms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids\lbphistograms', '.sqlite')

    # keep_only_pq_params_str(r'C:\data\computation\brodatz\pqcodes\histograms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\pqcodes\glcms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\pqcodes\lbphistograms', '.sqlite')

    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids_pairwise_distances\histograms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids_pairwise_distances\glcms', '.sqlite')
    # keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids_pairwise_distances\lbphistograms', '.sqlite')

    keep_only_pq_params_str(r'C:\data\computation\brodatz\centroids\sifts', '.sqlite')

