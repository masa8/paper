
##
### Input
### Outcome
### Output



## Multimodal Chain-of-Thought Reasoning in Language Models 
### Input
Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, Alex Smola
https://arxiv.org/pdf/2302.00923.pdf

### Outcome
Fix this:
  Existing CoT studies have focused on the language modality.
  
### Output
We propose 
    - Multimodal-CoT that incorporates language (text) and vision (images) modalities 
        - into a two-stage framework
            - rationale generation
            - answer inference. 
    - our model
        - outperforms the previous state-of-the-art LLM (GPT-3.5) 
            - by 16 percentage points (75.17%->91.68% accuracy) on the ScienceQA

        - Backbone: UnifiedQABase
        - INPUT: 
            - question text (Q),
            - context text (C),
            - multiple options (M)
            
        - OUTPUT:
            - A/Reasoning->A/Explaining->A/
            
        - Overview of our Multimodal-CoT framework. 
            - Multimodal-CoT consists of two stages: (i) rationale generation and (ii) answerinference. 
            - Both stages share the SAME model architecture but differ in the input and output. 
                - In the first stage, 
                    we feed the model with language and vision inputs to generate rationales. (QCM→R)
                - In the second stage, 
                    we append the original language input with the rationale generatedfrom the first stage. (QCMR→A)
                    Then, we feed the updated language input with the original vision input to the model to infer the answer.
                

## The MTA Dataset for Multi Target Multi Camera Pedestrian Tracking by Weighted Distance Aggregation
### Input
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w70/Kohl_The_MTA_Dataset_for_Multi-Target_Multi-Camera_Pedestrian_Tracking_by_Weighted_CVPRW_2020_paper.pdf
Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany
Philipp Ko ̈hl1,2

### Outcome
    Fix this:
        Because For now,
            - Existing multi target multi camera tracking (MTMCT) datasets are small
            - The creation of new real world datasets is hard as privacy has to be guaranteed and the labeling is tedious. 
            - Therefore a MTMCT dataset has been developed from Video Game.
         By creating a new large scale simulated dataset
         
### Output
    The system’s pipeline consists of stages for :
        - person detection, 
        - person re-identification, 
        - single camera multi target tracking, 
        - track distance calculation, 
            by :
            - a single camera time constraint
            - a multi camera time constraint using overlapping camera areas
            - an appearance feature distance
            - a homography matching with pair- wise camera homographies,
            - a linear prediction based on the velocity and the time difference of track

        - track association.
        
    Results
        we were able to surpass the results of state-of-the-art single camera trackers by +13% IDF1 score. 
        
        Person Detection   
            Approach        Trained on      AR      AP
            Faster R-CNN    COCO            14.6    11.3
            Faster R-CNN    MTA             64.8    61.6 ***
            Cascade R-CNN   MTA             69.5     67.0
    
        Person re-identification(market1501 dataset)
            Approach        Trained on    mAP    R-1
            AGW            MTA-ReID    33.7        64.0

        Tracking by ( based on Faster R-CNN Strong baseline)  on MTA dataset
        the core task is to track people across multiple cameras for as long as possible. 
            Identity metrics [30] best reflect this requirement
                 evaluation scores that by dividing the MTA test set into 10 parts with an approximate length of 5 min each 
                 then mean over all ten parts.
                 
            * Single camera tracking
                    Tracker            IDF1     IDP         IDR        IDs
                    IoU                 38.1     40.9     35.8        2370.3
                    DeepSORT        42.0     45.1     39.6        1797.8

            *  Multi camera tracking result
                    Configuration        IDF1        IDP        IDR        IDs
                    None            17.3         19.2        15.7        6869.5
                    All                30.1     33.6        27.3        7107.5
    
### Note
    IDF1 = 2 * (precision * recall) / (precision + IDF * recall)   
    F1 = 2 * (precision * recall) / (precision + recall)   
    the main difference between IDF1 and F1 is that IDF1 takes into account the rarity of the named entities in the text, whereas F1 only considers precision and recall. IDF1 is particularly useful when the named entities in the text are rare or have a skewed distribution, as it assigns more weight to the rare named entities and can result in a more balanced evaluation of the model's performance.    
    MTMCT stands for Multiple-Target Multi-Camera Tracking    
    
##Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
### Input
Karen Simonyan
Andrea Vedaldi
Andrew Zisserman
Visual Geometry Group, University of Oxford

### Outcome
 Fix the problem of ConvNet visualisation
### Output
We consider two visualisation techniques
 - The first one generates an image, which maximises the class score
    Sc(I) = w^{T}_{c} I + b_{c};
    w= ∂S_c/∂I | I_{0}
 - The second technique computes a class saliency map, specific to a given image and class
    Given an image I0 (with m rows and n columns) and a class c,
    class saliency map M ¥in R^{mxn}
    the derivative w is found by back-propagation
    M_{ij} = max_{c} |w_{h(i;j;c)}|
    

## On the Unreasonable Effectiveness of Centroids in Image Retrieval
### Input
Mikolaj Wieczorek∗
Barbara Rychalska
Jacek Dabrowski

### Outcome
        more robust to outliers
        more stable features
        both retrieval time and storage requirements are reduced significantly
  
### Output
    we propose to use the mean centroid representation both during training and retrieval
    
    We propose to use an aggregated item representation
    We propose the Centroid Triplet Loss (CTL). Instead of comparing the distance of an anchor image𝐴 to positive and negative instances, CTL measures the distance between 𝐴 and class centroids 𝑐𝑃 and 𝑐𝑁 representing either the same class as the anchor or a different class respectively.
    
    Outperforms the current state- of-the-art.        
    DeepFashion CTL-S-R50 CE
        mAP     0.404
        Acc@1   0.294
    Street2Shop CTL-L-R50IBN CE
        mAP     0.598
        Acc@1   0.537
    
## Leveraging sequential information from multivariate behavioral sensor data to predict the moment of calving in dairy cattle using deep learning
### Input
Arno Liseune a, Dirk Van den Poel a, Peter R. Hut c, Frank J.C.M. van Eerdenburg c, Miel Hostens b c
Faculty of Economics and Business Administration, Ghent University, Belgium
Faculty of Bioscience Engineering, Ghent University,Belgium
Faculty of Veterinary Medicine, Utrecht University,Netherlands
### Outcome
Calving is one of the most critical moments during the life of a cow and their calves.
Timely supervision is therefore crucial for animal welfare as well as the farm economics.

### Output
 Two sensors
    the neck and leg of each cow measured 
        rumination, 
        eating, 
        lying, 
        standup, 
        walking
        inactive behavior 
            on a minute basis.
 A deep learning model
 Results show that 
        calvings within 24 h    
            recall of 65%
            precision of 77%,
         calvings occurring within 3 h 
            recall of 57%
            precision of 49%
 Moreover, we find that using the missing value imputations significantly improves the predictive performance for observations containing up to 60% of missing values.
 
##Learn To Pay Attention
### Input
Saumya Jetley, Nicholas A. Lord, Namhoon Lee, Philip H.S. Torr
https://arxiv.org/abs/1804.02391

### Outcome
Fix this:
 CNN is largely inscrutable, impeding understanding of their successes and failures alike
 all attention map are implemented as post-hoc additions to fully trained networks
 
 
### Output
    Idea:
        we expect that enforcing a more focused and parsimonious use of image information should aid in generalisation over changes in the data distribution
    Estimator:
        Wee propose a trainable attention estimator
  
          The novelty of our contribution lies in 
            repurposing the global image representation 
                as a query to estimate
          we redesign standard architectures 
                such that they must classify the input image 
                    using only a weighted combination of local features, 
                        with the weights represented here 
                            by the attention map. 
                The network is thus forced to learn a pattern of attention relevant 
                    to solving the task at hand.

### Note
https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1


## Self-Supervised Pre-Training for Transformer-Based Person Re-Identification

### Input
    https://arxiv.org/abs/2111.12084
    Hao Luo, Pichao Wang, Yi Xu, Feng Ding, Yanxin Zhou, Fan Wang, Hao Li, Rong Jin
    
### Outcome
### Output

##ResT-ReID: Transformer block-based residual learning for person re-identification

### Input
https://www.sciencedirect.com/science/article/abs/pii/S016786552200085X

Ying Chen a b, Shixiong Xia a b, Jiaqi Zhao a b, Yong Zhou a b, Qiang Niu a b, Rui Yao a b, Dongjun Zhu a b, Dongjingdian Liu a b
School of Computer Science and Technology, China University of Mining and Technology, Xuzhou 221116, China
Engineering Research Center of Mine Digitization, Ministry of Education of the Peoples Republic of China, Xuzhou 221116, China

### Outcome
 Fix this:
    Transformer-based methods get robust representation but
    the memory and computational complexity cost vast overheads. 
  
### Output
ResT-ReID
 we use global self-attention in place of depth-wise convolution 
 we devise SIE-AGCN
 achieves competitive results compared with several state-of-the-art approaches

Train
 WRT Loss
 Center Loss
 ID Loss
 AR Loss
 
## Body Part-Based Representation Learning for Occluded Person Re-Identification
### Input
Vladimir Somers, Christophe De Vleeschouwer, Alexandre Alahi
https://github.com/lightas/Occluded-DukeMTMC-Dataset
    query images 
    gallery images
    train images
Dual Supervised Learning    
https://arxiv.org/pdf/1707.00415.pdf

### Outcome
Matching occluded person images with holistic ones

 
### Output
Part-based methods
 pros:  fine-grained information
        well suited to represent partially visible human bodies
 cons:  standard ReID is not adapted to local feature learning
        datasets are not provided with human topographical annotations

BPBreID https://github.com/VlSomers/bpbreid
        HIPPOCRATIC LICENSE
            MIT(INTELLECTUAL PROPERTY GRANTS) 
            Grant of Copyright License
            Grant of Patent License
            + 
            ETHICAL STANDARDS
                    [3.2.2.](#3.2.2) Provide equal pay for equal work
            SUPPLY CHAIN IMPACTED PARTIES
            NOTICE
            
        
Design two modules 
        body part attention module: predicting body part attention maps
            input: feature map
            output: ATTENTION map of K parts
            trained by : body part attention loss and human parsing labels
            
            pixel-wise part classifier
            body part attention loss
            human parsing labels.
                generated with the PifPaf　pose estimation model
                Human semantic regions are defined manually for a given value of K.
                {head, left/right arm, torso, left/right leg and left/right feet}.  K = 5.
                
            
        global-local representation learning module: producing holistic features
            input: feature map
            output: label
            trained by : ReID loss and label
            
Novel training scheme:  GiLt

Outperforms state-of-the-art methods by 
    0.7% mAP and 
    5.6% rank-1 accuracy 
    on the challenging Occluded-Duke dataset





