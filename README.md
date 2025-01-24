# Airbus-Aircraft-Detection
Aircraft detection using  a sample dataset from Airbus 
## Project description
The focus of this project is to compare the performance of a classical edge/contour detection technique with the performance of a modern Convolutional Neural Network (CNN) to detect aircraft at an airport. Using a sample dataset from Airbus' commerical satellite constellation, both techniques will be explored and the various factors that impact the techniques success will be analysed. Keeping track of all the aircraft at an airport can be difficult when a large amount of flights are frequently arriving and leaving the airport, and verifying each airplane is accounted for is vital,  Therefore, the testing and implementation of object detection technques on a dataset could a provide quick and effective method for aircraft detection. The results from the classical technique and CNN model are presented in the form of images with annotations depecting the locations of detected aircraft.

This project is composed of three main parts: Q1. Traditional Approach: A traditional non-neural network technique is used to explore the dataset, detect the desired features and highlight them. Q2. Traditional vs. Neural Network Approach: A PyTorch CNN is programmed to train on the dataset, detect the desired features and highlight them using bounding boxes based upon its confidence in the result. Q3. Impact of Neural Network Factors: Data augmentation and how it affects the perfomance of the CCN is explored.
### Technologies used
- **Edge/Contour detection:** OpenCV for classical edge detection methods.
- **Deep Learning:** Pytorch for programming the CNN.
- **Data Manipulation:** Pandas and NumPy for data conversion, Scikit learn for data splitting
- **Python Depenendcies:** Addtional dependencies can be found in `dependencies.txt`

This project comes with detailed explanation of the steps taken as is designed for beginners to learn about object detection algorithms and CNN's (Q1 & Q2), and research into how data augmentation affects CNN performance for intermediate users who have understood the content of the previous notebooks. 

## Why use a Classical approach?
Deep learning techniques have seen significant developments and improvements in recent years, and the use of neural networks has skyrockted in numerous applications from credit card fraud detection to semantic segmentation of football games. However, as neural networks have been developed, they have also become quite computationally expensive to run, require more expetise to implement effectively and usually require very large datasets to be trained adequately. These issues mean that classical techniques still have their uses in situations where data can be hard to obtain or when computation power is limited, if the problem can be solved using classical techniques effectively, it will require less computation power to run, less manpower to maintain and operate and be much faster to compute.

## Why use a Neural Network?
While classical techniques are usually faster and cheaper to implement, they can often struggle with unforseen complexities and are usually very rigid in their operation without outside assistence or modification for the new situation. Neural networks are much better at handling complex tasks and with a properly curated dataset, can still achieve the desired results despite the unseen data. Neural networks usually provide more accruate results compared to classical techniques as well, so when computation power isn't a restriction and the expertise is avaliable to implement it properly, neural networks are an excellent option.

## Dataset Overview
The Airbus sample dataset contains **109** images obtained by AirBus using their commercial satellite constellation, each image has a size of **2560 x 2560 pixels** (1280 x 1280 meters) and is stored in a JPEG format. The images were taken from various airports worldwide, some airports making repeat appearences at different acquisition dates and some images contain fog or clouds to diversify the dataset. **103** of the images have bounding boxes provided which depict the locations of the aircraft in each data image. The bounding box information is provided in `annoations.csv` and is composed of **4** columns, id, image id. geometry and class. **6** extra images do not have annotations and are designed to be used for evalutation of the trained models.

- **Total Rows:** 3246
- **Total Columns:** 4

### Column descriptions 

| Column    | Description     | Data Type   |
| ------- | ------------ | ------- |
| `id` | Unique id for each airplane | `int` |
| `image_id` | image the aircraft is located in | `str` |
| `geometry` | Four coordinate points on the image | `int` |
| `class` | Whether airplane is truncated or not | `str` |
### Sample Data
| id  | image_id                                    | geometry                                                                                  | class                |
| --- | ------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------- |
| 1   | 4f833867-273e-4d73-8bc3-cb2d9ceb54ef.jpg    | [(135, 522), (245, 522), (245, 600), (135, 600), (135, 522)]                              | Airplane            |
| 2   | 4f833867-273e-4d73-8bc3-cb2d9ceb54ef.jpg    | [(1025, 284), (1125, 284), (1125, 384), (1025, 384), (1025, 284)]                         | Airplane            |
| 80  | 2314c1b5-ec8f-4212-b42f-43365a13fd20.jpg    | [(1570, 0), (1695, 0), (1695, 73), (1570, 73), (1570, 0)]                                 | Truncated_airplane  |
| 81  | 2314c1b5-ec8f-4212-b42f-43365a13fd20.jpg    | [(1602, 2513), (1724, 2513), (1724, 2559), (1602, 2559), (1602, 2513)]                    | Truncated_airplane  |

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset/data)

## Dependency versions
- numpy = 2.2.1
- matplotlib = 3.10.0
- pandas = 2.2.3
- opencv-python  = 4.10.0.84
- pillow = 11.1.0
- torch = 2.5.1
- torchvision = 0.20.1
- sklearn = 1.6.0

## Installation and Usage
This repository can be cloned using
`https://github.com/LouieDormer/Airbus-Aircraft-Detection.git`
When running any of the notebooks, the following command can be run to install the necessary dependencies:
`!pip install -r dependencies.txt`
## License 
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.








