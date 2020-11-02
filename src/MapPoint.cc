/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapPoint.h"
#include "ORBmatcher.h"
#include <mutex>


namespace ORB_SLAM2 {

long unsigned int MapPoint::nNextId = 0;
/// 防止线程冲突的全局互斥锁
mutex MapPoint::mGlobalMutex;

/**
 * @brief 给定坐标与keyframe的MapPoint构造函数
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pRefKF KeyFrame
 * @param pMap   Map
 */
MapPoint::MapPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap) :         
    mnFirstKFid(pRefKF->mnId),                  // 创建该地图点的关键帧的id
    mnFirstFrame(pRefKF->mnFrameId),            // 创建该地图点的帧ID
    nObs(0),                                    // 被观测次数
    mnTrackReferenceForFrame(0),                // 防止被重复添加到局部地图点的标记
    mnLastFrameSeen(0),                         // 决定判断是否在某个帧视野中的变量
    mnBALocalForKF(0),                          // ?
    mnFuseCandidateForKF(0),                    // ?
    mnLoopPointForKF(0),                        // ?
    mnCorrectedByKF(0),                         // ?
    mnCorrectedReference(0),                    // ?
    mnBAGlobalForKF(0),                         // ?
    mpRefKF(pRefKF),                            // 参考关键帧,创建该地图点的关键帧
    mnVisible(1),                               // 在帧中的可视次数
    mnFound(1),                                 // 被找到的次数 和上面的相比要求能够匹配上
    mbBad(false),                               // 坏点标记
    mpReplaced(static_cast<MapPoint*>(nullptr)), // 替换掉当前地图点的点
    mfMinDistance(0),                           // 当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的下界
    mfMaxDistance(0),                           // 上界
    mpMap(pMap)                                 // 从属地图
{
    Pos.copyTo(mWorldPos);
    /// 平均观测方向初始化为0
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    /// 互斥量用于防止地图点编号冲突
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

/**
 * @brief 给定坐标与frame的MapPoint构造函数
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pMap   Map     
 * @param pFrame Frame
 * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
 */
MapPoint::MapPoint(const cv::Mat& Pos, Map* pMap, Frame* pFrame, const int& idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), 
    mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),
    mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0),
    mpRefKF(static_cast<KeyFrame*>(nullptr)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap) {
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    /// 世界坐标系下3D点到相机的向量 (当前关键帧的观测方向)
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);    // 归一化处理

    /// 计算深度范围
    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);    // 到相机的距离
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;
    mfMaxDistance = dist * levelScaleFactor;                              // 当前图层的"深度"
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];  // 该特征点上一个图层的"深度""

    /// 初始化最佳描述子 ToConfirm
    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    /// 防止不同线程创建地图点发生冲突
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

/// 设置地图点在世界坐标系下的坐标
void MapPoint::SetWorldPos(const cv::Mat& Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}
/// 获取地图点在世界坐标系下的坐标
cv::Mat MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

/// 世界坐标系下相机到3D点的向量 (当前关键帧的观测方向)
cv::Mat MapPoint::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}
/// 获取地图点的参考关键帧
KeyFrame* MapPoint::GetReferenceKeyFrame() {
     unique_lock<mutex> lock(mMutexFeatures);
     return mpRefKF;
}

/**
 * 添加观测信息,同时计算观测的相机数量,单目+1,双目+2
 * @param pKF 关键帧指针
 * @param idx 地图点在关键帧中对应特征点的索引
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    /// 如果已经存在观测,则忽略
    if (mObservations.count(pKF))
        return;
    /// 记录下能观测到该MapPoint的KF和在KF中的索引
    mObservations[pKF] = idx;

    if (pKF->mvuRight[idx] >= 0)
        nObs += 2;  /// 双目或者grbd +2
    else
        nObs++;     /// 单目 +1
}

/**
 * 删除观测关系 删除对应索引,相机数量,指针等
 * @param pKF  关键帧指针
 */
void MapPoint::EraseObservation(KeyFrame* pKF) {
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        /// 如果存在关联关系
        if (mObservations.count(pKF)) {
            int idx = mObservations[pKF];
            /// 减少观测相机计数
            if (pKF->mvuRight[idx] >= 0)
                nObs -= 2;
            else
                nObs--;
            /// 删除关键帧
            mObservations.erase(pKF);

            /// 如果该keyFrame是参考帧，重新指定RefFrame
            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            /// 当观测到该点的相机数目少于2时，丢弃该点
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        /// 设置可以删除该地图点的标识
        SetBadFlag();
}

/// 获取对当前点能够观测到关键帧
map<KeyFrame*, size_t> MapPoint::GetObservations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

/// 获取当前地图点的被观测次数
int MapPoint::Observations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

/// 删除地图点，并清除关键帧和地图中所有和该地图点对应的关联关系
void MapPoint::SetBadFlag() {
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for (map<KeyFrame*, size_t>::iterator mit = obs.begin(),
                 mend = obs.end(); mit != mend; mit++) {
        KeyFrame* pKF = mit->first;
        /// 把所有关键帧中对应该地图点的特征点索引位置清空为nullptr,表示该地图点被删除
        pKF->EraseMapPointMatch(mit->second);
    }
    /// 从地图集合中删除该地图点
    mpMap->EraseMapPoint(this);
}

/// 获取替换的地图点
MapPoint* MapPoint::GetReplaced() {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/// 替换地图点, 将this地图点替换为pMP,因为使用闭环后,需要重新建立地图点和关键帧的关系
void MapPoint::Replace(MapPoint* pMP) {
    /// 如果已经相同,则不需要替换
    if (pMP->mnId == this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        /// 暂存当前地图点的可视次数和被找到次数
        nvisible = mnVisible;
        nfound = mnFound;
        /// 保存替换点
        mpReplaced = pMP;
    }

    /// 所有能观测到该MapPoint的keyframe都要替换
    for (map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end();
         mit != mend; mit++) {
        KeyFrame* pKF = mit->first;
        if (!pMP->IsInKeyFrame(pKF)) { /// 如果替换的点不在关键帧的观测关系中
            pKF->ReplaceMapPointMatch(mit->second, pMP); // 替换关键帧特征点索引关系
            pMP->AddObservation(pKF, mit->second);       // 增加观测关系
        } else { /// 如果替换点在关键帧的观测关系中,就删除this_MapPoint对应关系
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    /// 将当前地图点的观测次数和被找到次数叠加到替换点上
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    /// 计算最匹配的描述子
    pMP->ComputeDistinctiveDescriptors();

    /// 删除当前地图点
    mpMap->EraseMapPoint(this);
}

/// 判断该地图点是否为Bad
bool MapPoint::isBad() {
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

/// 增加地图点被图像帧可视的次数, 可视不一定被找到被匹配
void MapPoint::IncreaseVisible(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

/// 增加地图点被图像帧中特征点匹配的次数
void MapPoint::IncreaseFound(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

/// 返回被找到匹配次数占可视的比例
float MapPoint::GetFoundRatio() {
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

/**
 * 计算最匹配的描述子
 *     一个MapPoint会被许多相机观测到，在插入关键帧后，需要判断是否更新当前点的最适合的描述子
 *     最好的描述子与其他描述子应该具有最小的平均距离
 *     可先获得当前点的所有描述子，然后计算描述子之间的两两距离，对所有距离取平均，得出最匹配描述子
 */
void MapPoint::ComputeDistinctiveDescriptors() {
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*, size_t> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());
    /// 遍历观测到该地图点的所有关键帧，获得orb描述子，并插入到vDescriptors中
    for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                 mend = observations.end(); mit != mend; mit++) {
        // mit->first: 观测到该地图点的关键帧
        // mit->second: 该地图点在关键帧中的索引
        KeyFrame* pKF = mit->first;

        if (!pKF->isBad())
            /// 对于可用的帧都提取对应的描述子并保存
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if (vDescriptors.empty())
        return;

    const size_t N = vDescriptors.size();
	
    /// 两两描述子的距离应该是对称的,所以可以三角计算即可
	std::vector<std::vector<float>> Distances;
	Distances.resize(N, vector<float>(N, 0));
	for (size_t i = 0; i < N; i++) {
        /// 和自己的距离是0
        Distances[i][i] = 0;
        for (size_t j = i+1;j < N;j++) {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    /// 选取最小的平均距离作为最匹配描述子
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i < N;i++) {
        /// 提取一行信息表示某个描述子与其他所有描述子的距离,进行排序
		vector<int> vDists(Distances[i].begin(), Distances[i].end());
		sort(vDists.begin(), vDists.end());

        /// 获取中值信息
        int median = vDists[0.5 * (N - 1)];
        
        /// 寻找最小的中值和对应描述子索引
        if(median < BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}

/// 获取当前地图点的描述子
cv::Mat MapPoint::GetDescriptor() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

/// 获取地图点在某个关键帧中对应的特征点ID, 没有置为-1
int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

/// 判断地图点是否在关键帧中
bool MapPoint::IsInKeyFrame(KeyFrame* pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * 更新法向量和深度值
 */
void MapPoint::UpdateNormalAndDepth() {
    map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        observations = mObservations;   /// 获得观测到该3d点的所有关键帧
        pRefKF = mpRefKF;               /// 观测到该点的参考关键帧
        Pos = mWorldPos.clone();        /// 3d点在世界坐标系中的位置
    }

    if (observations.empty())
        return;

    /// 初始值为0向量用于累加,每次累加后经过归一化
    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                 mend = observations.end(); mit != mend; mit++) {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        /// 相当于路标点坐标-相机坐标,表示观测方向
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali / cv::norm(normali);
        n++;
    }

    /// 深度范围：地图点到参考帧（只有一帧）相机中心距离，乘上参考帧中描述子获取金字塔放大尺度
    /// 得到最大距离mfMaxDistance;最大距离除以整个金字塔最高层的放大尺度得到最小距离mfMinDistance.
    /// 通常说来，距离较近的地图点，将在金字塔较高的地方提出，
    /// 距离较远的地图点，在金字塔层数较低的地方提取出（金字塔层数越低，分辨率越高，才能识别出远点）
    /// 因此，通过地图点的信息（主要对应描述子），我们可以获得该地图点对应的金字塔层级
    /// 从而预测该地图点在什么范围内能够被观测到
    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    /// 地图点到参照帧的距离
    const float dist = cv::norm(PC);
    /// 观测到该地图点的当前帧的特征点在金字塔的第几层
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    /// 当前金字塔层对应的缩放倍数
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    /// 金字塔层数
    const int nLevels = pRefKF->mnScaleLevels;
    {
        unique_lock<mutex> lock3(mMutexPos);
        /// 深度范围：地图点到参考帧（只有一帧）相机中心距离，乘上参考帧中描述子获取金字塔放大尺度
        mfMaxDistance = dist * levelScaleFactor;                              // 观测到该点的距离下限
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];    // 观测到该点的距离上限
        /// 平均观测方向
        mNormalVector = normal / n;
    }
}

/// 获取尺度变化范围的最小距离
float MapPoint::GetMinDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

/// 获取尺度变化范围的最大距离
float MapPoint::GetMaxDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

/// 预测尺度,预测特征点在金字塔的哪一层可以找到
/// 下图中横线的大小表示不同图层图像上的一个像素表示的真实物理空间中的大小
///              ____
/// Nearer      /____\     level:n-1 --> dmin
///            /______\                       d/dmin = 1.2^(n-1-m)
///           /________\   level:m   --> d
///          /__________\                     dmax/d = 1.2^m
/// Farther /____________\ level:0   --> dmax
///
///           log(dmax/d)
/// m = ceil(------------)
///            log(1.2)
/// 这个函数的作用:
/// 在进行投影匹配的时候会给定特征点的搜索范围,考虑到处于不同尺度(也就是距离相机远近,位于图像金字塔中不同图层)的特征点受到相机旋转的影响不同,
/// 因此会希望距离相机近的点的搜索范围更大一点,距离相机更远的点的搜索范围更小一点,所以要在这里,根据点到关键帧/帧的距离来估计它在当前的关键帧/帧中,
/// 会大概处于哪个尺度
int MapPoint::PredictScale(const float& currentDist, KeyFrame* pKF) {
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor 为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance / currentDist;
    }

    /// 同时取log线性化
    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

/**
 * @brief 根据地图点到光心的距离来预测一个类似特征金字塔的尺度
 * @param[in] currentDist       地图点到光心的距离
 * @param[in] pF                当前帧
 * @return int                  尺度
 */
int MapPoint::PredictScale(const float& currentDist, Frame* pF) {
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    /// 同时取log线性化
    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}
} //namespace ORB_SLAM
