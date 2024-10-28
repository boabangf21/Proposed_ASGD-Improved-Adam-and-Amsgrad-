
Abstract


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


Francis Boabang, ”Enhanced Stochastic Gradient Descent Algorithm for Machine Learning Training
and Its Application in Remote Surgical Gesture Recognition,” IEEE Transactions on Image Processing,
Revised based on the comments from IEEE internet of Things and resubmitted to IEEE Transactions
on Image Processing. Please, see the supplementary page for detail comments from IEEE Internet of
Things. Based on the recommendation of the IEEE Transactions on Image Processing, the paper has
been resubmitted to the IEEE Transactions on Neural Networks and Learning Systems.
