# Machine-vision
Assignments for CSC449

## Problem Solved
- initialize array: empty or zero </br>
  empty will have Nan which cause further problems
 
- boolean as integers</br>
  False 0</br>
  True 1
 
 - overlap of mask</br>
   A good way to calculate when no overlapping</br>
   mask = (x_region == np.max(x_region))
   
 - virtual environment</br>
   Use venv in pycharm terminal</br>
   
   `cd venv/bin</br>
   `source activate</br>
 
   `deactivate
  
 - wget in Mac
 
 `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" </br>
 
 `brew install wget
  
  ## Some Resources
  A Comprehensive Guide to Convolutional Neural Networks</br>
  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
  
  ml-cheatsheet</br>
  https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

  Stochastic Gradient Descent with momentum</br>
  https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
  
  PyTorch学习笔记(6)——DataLoader源代码剖析</br>
  https://blog.csdn.net/g11d111/article/details/81504637
  
  Fourier Transform? A visual introduction</br>
  https://www.youtube.com/watch?v=spUNpyF58BY
  
  Canny Edge Detection Step by Step in Python — Computer Vision</br>
  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
