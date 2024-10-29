# [CIKM'24] AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation

## Introduction

## This is the Pytorch implementation for our AlignGroup paper:

>AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation

## Environment Requirement
- python 3.9
- Pytorch 2.1.0

## Datasets

We use two public experimental datasets: **Mafengwo** and **CAMRa2011**. 
These two datasets' contents are in the `data/` folder.


## Run

```
# For Mafengwo 
python main.py --dataset=Mafengwo 

# For CAMRa2011 
python main.py --dataset=CAMRa2011 
```
For more running options, please refer to `main.py`



## Citation

If you find AlignGroup useful in your research or applications, please kindly cite:
```tex
@inproceedings{xu2024aligngroup,
  title={AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation},
  author={Xu, Jinfeng and Chen, Zheyu and Li, Jinze and Yang, Shuo and Wang, Hewei and Ngai, Edith CH},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={2682--2691},
  year={2024}
}
```

