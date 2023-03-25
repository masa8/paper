
##
### Input
### Outcome
### Output

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





