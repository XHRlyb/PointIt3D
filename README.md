# PointIt3D

Code and data for paper "**PointIt3D: A Benchmark Dataset and Baseline for** **Pointed Object Detection Task** ", Chunru Lin, Hongxin Zhang and Haotian Zheng.



## Dataset

Our PointIt3D dataset is available at [BaiduPan](https://pan.baidu.com/share/init?surl=E3u96E7dEXnrR1dDris_1w)(access code: jps5)



## Baseline

Download PointIt3D Dataset as above and unzip to prepare the dataset like this:

````
```
PointIt3D
├── ScanNet_with_eric
|   ├── scene0000_00_with_eric
|   |   ├── answer.txt
|   |   ├── scene0000_00.aggregation.json
|   |   ├── scene0000_00_vh_clean_2.0.010000.segs.json
|   |   ├── scene0000_00_with_eric.ply
|   ├── scene0000_01_with_eric
|   ├── ...
|   ├──scannetv2-labels.combined.tsv
```
````

Then run below to get baseline running

```bash
cd baseline
python3 demo3.py
```

