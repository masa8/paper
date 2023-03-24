
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





