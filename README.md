## LENET-5 :</br>
__Team Name__ – Outliers</br>
__Team Number__ – 13</br>
</br>

__Members:__ &nbsp; &nbsp; Suvadeep Maiti - 2021702021</br>
&emsp; &emsp;&emsp; &emsp; &emsp;Praguna Manvi - 2021701031</br>
&emsp; &emsp;&emsp; &emsp; &emsp;Haasith Pasala - 2021702017</br>
&emsp; &emsp;&emsp; &emsp; &emsp;Laksh Nanwani - 2021701002</br>

</br>

### Problem Statement: </br>

Implementing Lenet-5, a simple and straighforward convolutional neural network architecture with gradient-based learning, for recognizing handwritten digits, from scratch. Lenet-5 comprises of 7 layers, not counting the input, all of which contains trainable parameters(weights). It takes an input of 32x32 pixel image and outputs the likelihood of the digit present in the image.

<img src="images/lenet.png" alt="Lenet-5" />

### Project objectives:</br>
• Build Lenet – 5 from scratch using basic libraries such as NumPy.</br>
• Evaluate with inbuilt Lenet – 5 model trained with libraries.</br>
• Achieve an accuracy close to the paper’s on MNSIT test set.</br>
### Deliverables:</br>
• Preprocess MNSIT Dataset containing 60000 samples.</br>
• Implementing hidden layers and forward pass.</br>
• Implementing back-propagation from scratch.</br>
• Visualizing and comparing the results on MNSIT data set.</br>
### Code Structure
------------------

    ├── MNIST                         
    ├── images                    
    ├── src  
    ├   ├── Results
    ├   ├   ├── #plot images      
    ├   ├── notebooks 
    ├   ├   ├── Lenet.ipynb
    ├   ├   ├── model-relu-tanh-28-512-adam.pickle
    ├   ├── scripts 
    ├   ├   ├── RBF_init.py
    ├   ├   ├── model.py
    ├   ├── MNIST_auto_Download.py                          
    └── README.md
-----------
</br>
In the above structure, the source code is found in the 'src' directory in 'Lenet.ipynb' file. This notebook book file has
</br>
</br>

### Pre-requisites:
 
Before running the code, following python libraries are to be installed.

------------------
numpy  
opencv  
afsfa  
   
-----------
</br>

### Dataset:

We used the Modified NIST (MNIST) dataset which is a subset of the NIST database. It is a database of handwritten digits with a training set of 60,000 samples and a test set of 10,000 samples. This dataset is downloaded using MNIST_auto_Download.py file (found online).</br>
Ref: http://yann.lecun.com/exdb/mnist/
</br>
</br>

### Timeline (Year 2021):

</br>

| Timeline | Milestone |
| ------------- | ------------- |
| 26<sup>th</sup> Oct  | Project Allocation  |
| 7<sup>th</sup> Nov  | Project Proposal Submission  |
| 8<sup>th</sup> Nov - 11<sup>th</sup> Nov  | Paper and relevant work reading  |
| 12<sup>th</sup> Nov – 17<sup>th</sup> Nov  | Implementing hidden layers and forward pass |
| 18<sup>th</sup> Nov – 20<sup>th</sup> Nov  | Mid evaluation  |
| 21<sup>st</sup> Nov – 27<sup>th</sup> Nov  | Backpropagation implementation  |
| 27<sup>th</sup> Nov – 3<sup>rd</sup> Dec  | Training, testing, Analysis and ppt preparation  |
| 4<sup>th</sup> Dec  | Final Evaluation  |

</br>

### Work Distribution:</br>
</br>

| Member | Tasks |
| ------------- | ------------- |
| Suvadeep | Preprocessing, Convolution - Forward & backward |
| Praguna  | Training, Sub-sampling, Combining layers |
| Laksh | Testing, Full connection - Forward & backward |
| Haasith | PPT, Gaussian connections - Forward & backward  |

</br>

### References:</br>
1. http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf</br>
