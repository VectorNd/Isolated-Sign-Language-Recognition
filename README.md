# Isolated Sign Language Recognition

## Objective   

- To classify Isolated American Sign Language Signs. 
- This work can help improve the ability of Popsign games and help the relatives of deaf children learn basic signs and communicate better with their loved ones. 

## Dataset 

- **Landmarks (such as Pose , Left Hand , Lips) co-ordinates data** extracted from the raw processed videos using **Mediapipe Model**. 
- Dataset contains **250 unique signs**. 
- **21 unique participants** contributed in the building of the dataset as per the participant_id provided in the data.
- **Dataset** :  https://www.kaggle.com/competitions/asl-signs/data

## Evaluation Metric 

- **Simple Classification Accuracy**

## Approach 

- In the Data provided , for each frame in a video , Co-ordinates of Landmarks (such as Hand , Lips , Pose,etc.) keypoints are provided and extracted using Mediapipe Model.
- For each frame , Right / left Half portion keypoints and Lips keypoints are extracted out from the data . Only One half (i.e. Right or Left) keypoints and Lips Keypoints (i.e. Total 66 keypoints) are used at a time because after experimentation and other participant's comments , it was found out that using both halfs simultaneously is confusing the model and it does not improves the LB or CV Score . Half with more nan values is discarded out .
- Frames having nan values for the more dominant half are discarded out. Each sequence of frames is of variable length . To combat this , INPUT_SIZE = 64 is chosen on the basis of histogram of non-empty frames in a single sequence .
- For shorter sequences of frames, padding with zeros is done to match the INPUT_SIZE and masking is done  to avoid them during Training.
- For longer sequences of frames , padding is done from both left and right side to make the length a multiple of INPUT_SIZE and then mean pooling is done to not lose the information and match the INPUT_SIZE.
- Finally Transformer is written from scratch in Tensorflow so that the whole pipeline (ie. TFLite Model) can run on a mobile or pc locally.
- The embedding layer makes an embedding per landmark(lips/left hand/right hand/arm pose) and merges these embedding with fully connected layers. The transformer consists of just 2 blocks with a simple mean pooling and fully connected layers for classification.

## Results :

- **Accuracy Score** : 0.99 on Training Data 
- **CV Score** : 0.76

## Notebook Links :

- https://www.kaggle.com/code/rishabhkamboj2003/transformer-training

