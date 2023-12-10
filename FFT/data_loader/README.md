## Data Loader                                                                                 

Reading data from a specified path (root path at ./Data folder) as an Dataset class object

- 1D Data (finger data)
    - Converted from Matlab format (Column-Major)
    - Indices starting from 1
    - Data loaded as a 1D array with proper Accessors and Setters
    - Dimensions:
        - Rows : index of channel
        - Cols : time steps
        - Depth: number of trials for each channel
- 2D Data (CIFAR10)
    - Indices starting from 0
    - Data loaded as  
        - RGB values of 1D array (Row-Major)
        - Converted to grayscale image with pixel values ranging from 0 to 255 
    - Dimensions:
        - Rows      : Height
        - Cols      : Width
        - Channels  : Channels
        - numImages : number of images
