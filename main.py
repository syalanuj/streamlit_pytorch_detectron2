import streamlit as st
import torch,torchvision
# import req
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random

print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")
setup_logger()

st.title('Deploying a Pytorch model on Streamlit | Detectron2')
st.write('What is Detectron2?')
st.write('Detectron2 is Facebook AI Researchs next generation software system that implements state-of-the-art object detection algorithms. It is a ground-up rewrite of the previous version, Detectron, and it originates from maskrcnn-benchmark.')
st.image('assets/img.png')