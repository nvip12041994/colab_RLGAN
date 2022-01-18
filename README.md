# colab_RLGAN
https://colab.research.google.com/drive/1IU8Y_gAdsxVVwimhEGSCpJI26wHEykmK#scrollTo=qbt7LXX6G6SK

# online server
export CUDA_HOME=/usr/local/cuda
pip install --editable ./

# to install lightconv
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install

# to install dynamicconv
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install