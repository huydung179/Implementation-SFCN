CUDA_VISIBLE_DEVICES = "1"

EPOCHS = 80

T1_DIR = '/opt/deep/data/datasets/lifespan1_CN/CN/t1'

TRAIN_DIR = '/opt/deep/data/resized_input'

METADATA_CSV = '/opt/deep/data/csv/metadata/all_available_data/output/Lifespan1_CN.csv'

# T1_TEST_DIR = '/opt/deep/data/datasets/common_test_age_prediction_out_of_lifespan/CN/t1'

TEST_DIR = '/opt/deep/data/resized_input_test'

METADATA_CSV_TEST = '/opt/deep/data/csv/metadata/all_available_data/output/all_cn.csv'
# METADATA_CSV_TEST = '/opt/deep/data/csv/metadata/all_available_data/output/common_test_age_prediction_out_of_lifespan.csv'

OUTPUT_NETWORK = '/opt/deep/data/all_configs/reimplement_papers/han_peng_2021/models/'

OUTPUT_EVAL = '/opt/deep/data/output'