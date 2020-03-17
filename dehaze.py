import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_list = glob.glob("test_images/*")
	dehaze_net = net.dehaze_net().to(device)
	dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))
	for image in test_list:
		data_hazy = Image.open(image)
		data_hazy = (np.asarray(data_hazy)/255.0)

		data_hazy = torch.from_numpy(data_hazy).float()
		data_hazy = data_hazy.permute(2,0,1)
		data_hazy = data_hazy.to(device).unsqueeze(0)
		clean_image = dehaze_net(data_hazy)
		torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image.split("/")[-1])
		print(image, "done!")
