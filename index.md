## PointIt3D: A Benchmark Dataset and Baseline for Pointed Object Detection Task

> PointIt3D: A Benchmark Dataset and Baseline for Pointed Object Detection Task 
>
> Chunru Lin, Hongxin Zhang, Haotian Zheng
>
> Shanghai Jiao Tong University

<img src="https://xhrlyb.github.io/PointIt3D/images/1.a.png" alt="img" style="zoom: 47%;" />
![img](https://xhrlyb.github.io/PointIt3D/images/1.b.png)

### Abstract

Pointed object detection is of great importance for human-machine interaction, but attempts to solve this task may run into the difficulties of lack of available large scale datasets since people hardly record 3D scenes with a human pointing at specific objects. In efforts to mitigate this gap, we cultivate the first benchmark dataset for this task: PointIt3D, containing 347 scans now and can be easily scaled up to facilitate future utilizations, which is automatically constructed from existing 3D scenes from ScanNet[1] and 3D people models using our novel synthetic algorithm that achieves a high acceptable rate of more than 85% according to three experts’ assessments, which hopefully would pave the way for further studies. We also provide a simple yet effective baseline based on anomaly detection and majority voting pointline generation to solve this task based on our dataset, which achieves accuracy of 55.33%, leaving much room for further improvements. Code will be released at [https://github.com/XHRlyb/PointIt3D](https://github.com/XHRlyb/PointIt3D).

### Video

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/iCbW0HwWO3k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Code

Our source code is available on [Github](https://github.com/XHRlyb/PointIt3D)

### PointIt3D Dataset

Our PointIt3D dataset is available at [BaiduPan](https://pan.baidu.com/share/init?surl=E3u96E7dEXnrR1dDris_1w)(access code: jps5)

### Citation

```
@article{Pointed3D
    author = {Chunru Lin, Hongxin Zhang, Haotian Zheng},
    title = {PointIt3D: A Benchmark Dataset and Baseline for Pointed Object Detection Task},
    year = {2021},
    howpublished={https://xhrlyb.github.io/PointIt3D/}
}
```
