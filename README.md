# csc490-project

🖼️ Image Inpainting with Knowledge Distillation
CSC490 - Computer Vision with Machine Learning
Group 7 - Behrouz Akhbari

🔍 Overview
Image inpainting is the process of reconstructing missing parts of an image. This technique is essential in various applications such as:

Photo restoration

Object removal

AI-generated content

The main goal of this project is to build a computationally efficient inpainting model that maintains high-quality output, suitable for resource-constrained environments.

🚀 Key Innovation: Knowledge Distillation
We leverage knowledge distillation to train a smaller "student" model to replicate the performance of a larger, pre-trained "teacher" model.
✅ Benefits:

Reduced computational cost

Maintained visual fidelity

🗂️ Dataset: Places2
We use the Places2 dataset due to its:

High-resolution images

Diverse indoor and outdoor scenes

Rich variation in textures, lighting, and object placements

Dataset Preparation
Sampling: 10,000 images selected with diverse occlusions and scenes.

Masking: Object-aware masks generated using Mask R-CNN.

If no object is detected, a random mask is generated.

# Examples of our work 

![Untitled design](https://github.com/user-attachments/assets/d4840948-7982-42ec-b88f-5ef9d032b888)
![Untitled design (1)](https://github.com/user-attachments/assets/4eee537a-608d-482a-a527-d70cdaf56494)


As you can see this model is substantially smaller than the Big-Lama model (which contains 51 million parameters) with only about 18 million parameters and has only been trained for 30 epochs on a dataset of about 20000 images and it works relatively well for a 256x256 image.
Furthermore, it does work for higher resolution images as is evident in the second example.


