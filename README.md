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
  
  ## Some Resources
  A Comprehensive Guide to Convolutional Neural Networks</br>
  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
