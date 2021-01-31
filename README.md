# Machine-vision
Notes for CSC449. </br>
More in the following two repos:</br>
  Hough transformation https://github.com/ChloeZPan/Hough-Transform</br>
  DCF tracker https://github.com/ChloeZPan/DCF-tracker
  
Due to the course policy, I cannot post my code.

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
   https://zhuanlan.zhihu.com/p/60647332
   
   `cd venv/bin`</br>
   `source activate`</br>
   `deactivate`
  
    conda env</br>
    https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
    
    `cd miniconda3/bin`</br>
    `source activate`</br>  
    `conda init`
  
 - wget in Mac
 
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`</br>

   `brew install wget`
  
  - download file from google drive
  
    `pip install gdown`</br>
    `gdown https://drive.google.com/uc?id=file_id`
    
    
    `gdown https://drive.google.com/uc?id=1ZqJs2AGtMTfGTgH1oHTP_ZEFXRNX3uLj`

  - permission denied when run ./get_stanford_models.sh
  
    `chmod u+x get_stanford_models.sh`</br>
    `./get_stanford_models.sh`
  
  - _tkinter.TclError: no display name and no $DISPLAY environment variable</br>
    https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
    
    `import matplotlib`</br>
    `matplotlib.use('Agg')`

  ## Some Resources
  A Comprehensive Guide to Convolutional Neural Networks</br>
  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
  
  ml-cheatsheet</br>
  https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

  Stochastic Gradient Descent with momentum</br>
  https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
  
  PyTorch - DataLoader</br>
  https://blog.csdn.net/g11d111/article/details/81504637
  
  Fourier Transform? A visual introduction</br>
  https://www.youtube.com/watch?v=spUNpyF58BY
  
  Canny Edge Detection Step by Step in Python — Computer Vision</br>
  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
  
  Matrix derivatives</br>
  https://blog.csdn.net/promisejia/article/details/80159619
  
  Properties of the Trace and Matrix Derivatives</br>
  https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf

  Pytorch - Word Embedding</br>
  https://www.pytorchtutorial.com/10-minute-pytorch-7/
  
  Pytroch - pack_padded_sequence()</br>
  https://www.cnblogs.com/sbj123456789/p/9834018.html</br>
  https://blog.csdn.net/lssc4205/article/details/79474735
  
  N-gram</br>
  https://blog.csdn.net/songbinxu/article/details/80209197
  
  LSTM</br>
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  
  Image Captioning (CNN-RNN)</br>
  https://shenxiaohai.me/2018/10/22/pytorch-tutorial-advanced-04/
  
  Embed attention mechanism into image cpation model</br>
  https://www.jianshu.com/p/79f48437a442
  
  Input and output dimension in rnn,lstm,gru</br>
  https://www.jianshu.com/p/b942e65cb0a3
  
  Pytorch from scratch</br>
  https://baijiahao.baidu.com/s?id=1597446499634684834&wfr=spider&for=pc
