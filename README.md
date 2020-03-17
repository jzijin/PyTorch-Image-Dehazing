# PyTorch-Image-Dehazing
PyTorch implementation of some single image dehazing networks. 

Currently Implemented:
**AOD-Net**: An extremely lightweight model (< 10 KB). Results are good.


**Prerequisites:**
1. Python 3 
2. Pytorch 0.4

**Preparation:**
1. Create folder "data".
2. Download and extract the dataset into "data".
- training_images:[http://39.101.192.22/training_images.zip](http://39.101.192.22/training_images.zip)
- original_images:[http://39.101.192.22/original_image.zip](http://39.101.192.22/original_image.zip)

**Training:**
1. Run train.py. The script will automatically dump some validation results into the "samples" folder after every epoch. The model snapshots are dumped in the "snapshots" folder. 

**Testing:**
1. Run dehaze.py. The script takes images in the "test_images" folder and dumps the dehazed images into the "results" folder. A pre-trained snapshot has been provided in the snapshots folder.

**Evaluation:**
WIP.  
Some qualitative results are shown below:

![Alt text](results/man.png?raw=true "Title")  
![Alt text](results/guogong.png?raw=true "Title")  
![Alt text](results/test4.jpg?raw=true "Title")  
![Alt text](results/test9.jpg?raw=true "Title")  
![Alt text](results/test13.jpg?raw=true "Title")  
![Alt text](results/test15.jpg?raw=true "Title")
