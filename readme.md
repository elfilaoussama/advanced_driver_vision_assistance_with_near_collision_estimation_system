# Advanced Driver Vision Assistance with Near Collision Estimation System

This repository contains a **real-time object distance estimation system** designed to assist in advanced driver assistance systems (ADAS). The system utilizes computer vision techniques, including object detection and depth estimation, to predict vehicle and object distances in real-time. By using cloud-based resources for processing, this system reduces hardware requirements on-board the vehicle, making it more cost-effective, scalable, and adaptable to a variety of vehicular platforms.

## Project Overview

In traditional vehicle distance estimation systems, embedded hardware is used to process data directly within the vehicle. However, such systems often have high costs, limitations in scalability, and significant hardware requirements. In this project, we aim to overcome these limitations by adapting the framework of **"Vehicle Distance Estimation from a Monocular Camera for Advanced Driver Assistance Systems"** into a cloud-based solution.

By transitioning to a **client-server architecture**, we leverage cloud resources to process images and predict distances, significantly reducing the hardware needed on the vehicle itself. This shift enables the solution to be more scalable and adaptable to a wide range of vehicles, making it easier to deploy across different platforms while maintaining real-time performance.

The system integrates multiple models, including **DETR** for object detection, **GLPN** for depth estimation, and **LSTM** for predicting historical data. With this approach, we have created the **Advanced Driver Vision Assistance with Near Collision Estimation System** (ADVANCE). The main innovations of this system include its ability to perform real-time distance prediction with low-cost infrastructure and its flexibility in using cloud resources, ensuring broad accessibility and reduced hardware dependencies.

## Credits

This project builds on the work from [KyujinHan's Object-Depth-Detection-based-Hybrid-Distance-Estimator](https://github.com/KyujinHan/Object-Depth-detection-based-hybrid-Distance-estimator). Their original implementation served as an important starting point for this project, where we extended their ideas into a cloud-based architecture to reduce embedded system dependency and enable real-time, scalable distance estimation.

## Requirements

Before running the application, ensure you have all the necessary dependencies installed.

### Install Dependencies

If you have a `requirements.txt` file, you can install the required Python packages using the following command:

```bash
pip install -r requirements.txt 
```

### Key Additions:
1. **Rebuild Cython Extension**:
   - Added the command `python setup.py build_ext --inplace` in the **Setting Up and Running the Application** section to ensure the Cython extension is built correctly when setting up the project in a new environment or after modifying Cython files.

