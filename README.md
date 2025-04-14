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

Folder Structure
sql
Copy
Edit
Target/ → Original images  
Mask/   → Binary masks indicating occluded areas  
Label/  → Images with masked areas whited out  
💡 Why This Approach Works
Creates realistic and diverse occlusions for robust training.

Maintains a well-organized dataset structure for reproducibility and scalability.

📚 References
Knowledge Distillation Illustration – Neptune AI

Dogra et al., Exploring image inpainting for seamless restitution (Laser Focus World, 2023)

Prakya et al., Photobombing removal benchmarking, Advances in Visual Computing, 2022.

Let me know if you'd like a version with code examples, model architecture, or usage instructions added in!
