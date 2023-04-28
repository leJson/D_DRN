This repository contains the source code for the research paper titled "Towards End-to-End Deep RNN based Networks to Precisely Regress of the Lettuce Plant Height by Single Perspective Sparse 3D Point Cloud". The paper presents four models: Z-B, BaseLine, DRN, and D-DRN. These models are designed for the precise regression of lettuce plant height using a single perspective sparse 3D point cloud.


Main env installation:

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

More env information can be found in requirements.txt.

To execute the models, please run the following commands in your terminal:

./Z_B.sh
./BaseLine.sh
./DRN.sh
./D_DRN.sh
The results of the execution will be stored in a file named "result". Please note that the performance of each model may vary based on the input data and the hardware configuration.
