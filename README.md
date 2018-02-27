# mot_evaluation
It is a python implementation of [MOT](https://motchallenge.net/). However, I only reimplement the 2D evaluation part under MOT16 file format.

### Metrics
The metrics of MOT16 are based on the following papers:

1. CLEAR MOT
- Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." Journal on Image and Video Processing 2008 (2008): 1.
2. MTMC
- Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European Conference on Computer Vision. Springer, Cham, 2016.

Typical evaluation format is shown as
```bash
IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
```
The meaning of each alias is 
- **IDF1(ID F1 Score)**:
- **IDP(ID Precison)**: 
- **IDR(ID Recall)**
- **Rcll(Recall)**
- **Prcn(Precision)**
- **FAR(False Alarm Ratio)**
- **GT(# Groundtruth Trajectory)**
- **MT(# Mostly Tracked Trajectory)**
- **PT(# Partially Tracked Trajectory)**
- **ML(# Mostly Lost Trajectory)**
- **FP(# False Positives)**
- **FN(# False Negatives)**
- **IDs(# IDSwitch)**
- **FM(# Fragmentations)**
- **MOTA**
- **MOTP**
- **MOTAL(MOTA Log)**

### To Do
- Debug 
