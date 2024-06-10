# Introduction

This file describes how to use the datasets used in:

**Benefit of Large Field-of-View Cameras for Visual Odometry** \
Z. Zhang, H. Rebecq, C. Forster and D. Scaramuzza (ICRA 2016) \
[Paper](http://rpg.ifi.uzh.ch/docs/ICRA16_Zhang.pdf) \
[Video (Youtube)](https://youtu.be/6KXBoprGaR0)

The following sections describe where to get the ground truth information (camera calibration, ground truth trajectory and depth), and how to interpret it.

If you use this dataset in research work, we kindly ask you to cite the following:

    @inproceedings{Zhang2016ICRA,
      title={Benefit of Large Field-of-View Cameras for Visual Odometry},
      author={Zhang, Zichao and Rebecq, Henri and Forster, Christian and Scaramuzza, Davide},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
      year={2016},
      organization={IEEE}
    }

For any question or bug report regarding this dataset, please send an e-mail to zzhang (at) ifi (dot) uzh (dot) ch. If you are interested in rendering your own catadioptric datasets, please send an e-mail to rebecq (at) ifi (dot) uzh (dot) ch .

# How to use the datasets?

## Camera intrinsics

The intrinsics are given in the file *intrinsics.txt*.

Note that the camera model used differs depending on the field-of-view (FoV). For the perspective camera, the model used is the pinhole camera model (thus we provide the calibration matrix K), whereas for the fisheye and catadioptric datasets, the camera model used is the omnidirectional camera model (https://sites.google.com/site/scarabotix/ocamcalib-toolbox). A C++ implementation of this camera model (forward projection and back-projection functions) is available at: https://github.com/zhangzichao/omni_cam . 

## Ground truth trajectory

The file *images.txt* defines a set of associations between an image number (image_id) and its corresponding timestamp (in seconds). The file *groundtruth.txt* contains the camera poses for each image id.

### Format of *images.txt*

    image_id timestamp (seconds) path_to_img


### Format of *groundtruth.txt*

The trajectory is defined as a sequence of poses (given in translation + quaternion representation), which define the transformation T_world_camera: from camera coordinates to world coordinates (transforms a point in camera frame to world frame).

    image_id tx ty tz qx qy qz qw

## Ground truth depth maps

This only applies if you downloaded the datasets that contain also the depth maps. The files **.depth* contain the raw depth values for each pixel in the image. The depths are stored in row major order. The size is the same as the images. The depths are encoded as floating point values and separated by a space.

### Format of *depthmaps.txt*

    image_id path_to_img

#### IMPORTANT NOTE

There are two ways to encode the depth of a 3D point P (in the camera coordinate frame). The first one is to use the Z coordinate of P (depth along the Z axis = optical axis). The second one is to use the distance along the ray passing through point P (euclidean distance along the optical ray). Denoting by f the (unit norm) bearing vector corresponding to point P (and f_z its z component), you can convert between the two representations using:

    depth_z = depth_euclid * f_z .

The depths provided in the "*.depth" files correspond to distances along the optical ray (the second representation depth_euclid).

## Miscellaneous notes
 - In Blender, the render engine used was Cycles (it is a ray-tracing engine)
 - For the fisheye and catadioptric datasets, we used our implementation of the omnidirectional camera model in Blender, which is available at: https://github.com/uzh-rpg/rpg_blender_omni_camera
 - For the pinhole/perspective dataset, we used the default perspective Blender camera
 
 ## Notes about the urban canyon dataset
 - The Blender addon Scene City (http://cgchan.com/store/scenecity) was used to generate the city 3D model.
 - The dataset is mainly intended to compare the performance of visual odometry (VO) algorithms in a (geometrically) realistic scenario. It was not intended for perfect photo-realism, although the images are quite plausible, thanks to the ray-tracing engine used.
