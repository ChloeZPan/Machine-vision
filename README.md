# Machine-vision
Assignments for CSC449

# Problem Solved
- initialize array: empty or zero
  empty will have Nan which cause further problems
 
- boolean as integers
  False 0
  True 1
 
 - overlap of mask
   A good way to calculate when no overlapping
   mask = (x_region == np.max(x_region))
  
