
**Abstract**


The recognition of surgical gestures holds significant
importance for enhancing the intelligence of next-generation
robot-assisted or remote robotic surgery systems. To associate specific surgical gestures with corresponding surgical tools, a learning
phase is important. Adaptive stochastic gradient descent (ASGD)
has emerged as a robust framework for optimizing machine
learning (ML) algorithms in various recognition applications, such
as surgical gesture recognition and speech recognition. However,
the task of surgical gesture recognition involves features with
both large and small values, posing a challenge for learning
with ASGD due to the small learning rate dilemma. To tackle
this concern, we propose a novel ASGD approach that leverages
the non-uniform p-norm concept to assign distinct categories of
coordinate values with different base learning rates. Additionally,
we provide theoretical guarantees for the efficacy of the proposed
ASGD method in convex and nonconvex settings. Subsequently,
we assess the performance of our method on the CIFAR-100
dataset by comparing it with state-of-the-art optimizers in the
fields of machine learning and computer vision. Finally, we
demonstrate the ability of our proposed ASGD approach to detect
suturing gestures within the surgical gesture recognition task by
benchmarking it with the state-of-the-art optimizers. The results
illustrate that our proposed optimizer surpasses state-of-the-art
approaches in terms of generalization performance.


**Citation** 

Francis Boabang, ”Enhanced Stochastic Gradient Descent Algorithm for Machine Learning Training
and Its Application in Remote Surgical Gesture Recognition,” under review at 
IEEE Transactions on Neural Networks and Learning Systems.

**Related Work**


H. Huynhnguyen and U. A. Buy, "Toward Gesture Recognition in Robot-Assisted Surgical Procedures," 2020 2nd International Conference on Societal Automation (SA), Funchal, Portugal, 2021, pp. 1-4,

Chen, Jinghui, et al. "Closing the generalization gap of adaptive gradient methods in training deep neural networks." arXiv preprint arXiv:1806.06763 (2018).



**Prerequisites**:
Tensorflow version 2.13.1


