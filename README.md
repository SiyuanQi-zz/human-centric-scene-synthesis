# CVPR2018 - Human-centric Indoor Scene Synthesis Using Stochastic Grammar

## Running in Virtual Machine
Tested with VMWare workstation 14.1.1. VirtualBox should also work
https://drive.google.com/file/d/12hRpv07lvorx9-Yv6fSrMPAECFJlxWHm/view?usp=sharing
Login password: cvpr


## Compile from Source
### Requirements
Tested in Ubuntu 16.04
- OpenCV, Boost library, CGAL, Eigen
```
sudo apt-get install libopencv-dev libcgal-dev libcgal-qt5-dev libboost-all-dev libeigen3-dev
```

### Running the sampler
- Download the following file and save to folder `src/metatdata/`
    - `stats`: https://drive.google.com/file/d/1eTv3QlSqa1Pq9Bm94nbTB0dMwErxn3Ou/view?usp=sharing
- Modify the paths in the `main` function in `src/cpp/main.cpp`
- Build and run the sampler:
```cpp
cd src/cpp
mkdir build && cd build
cmake ..
make -j8
./cpp/cpp 0
```
- The sampled results are saved as txt and json files in `tmp/samples'. The txt files can be opened and viewed by [RoomArranger](http://www.roomarranger.com/) (free to use on Linux).

### Visualizing the results
- Change the paths in `src/python/config.py`
- Run `src/python/visualize_activity.py`. The visualization results will be saved in 'tmp/figures'.


If you find this code useful, please cite our work with the following bibtex:
```
@inproceedings{qi2018human,
    title={Human-centric Indoor Scene Synthesis Using Stochastic Grammar},
    author={Qi, Siyuan and Zhu, Yixin and Huang, Siyuan and Jiang, Chenfanfu and Zhu, Song-Chun},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
