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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <mutex>

namespace ORB_SLAM2 {

/// 关键帧的全局帧号
long unsigned int KeyFrame::nNextId = 0;

/// 关键帧的构造函数
KeyFrame::KeyFrame(Frame& F, Map* pMap, KeyFrameDatabase* pKFDB) :
    mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS),
    mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv),
    mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0), 
    mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnLoopQuery(0), 
    mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), 
    mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight), mvDepth(F.mvDepth), 
    mDescriptors(F.mDescriptors.clone()), mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), 
    mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor), 
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
    mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), 
    mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), 
    mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB), 
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true),
    mpParent(nullptr), mbNotErase(false), mbToBeErased(false), mbBad(false),
    mHalfBaseline(F.mb / 2), mpMap(pMap) {

    /// 获取id
    mnId = nNextId++;
    /// 根据当前帧列数给关键帧列数赋值
    mGrid.resize(mnGridCols);
    /// 把当前帧栅格信息拷贝到关键帧上
    for (int i = 0; i < mnGridCols; i++) {
        mGrid[i].resize(mnGridRows);
        for(int j = 0; j < mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }
    /// 将当前帧的姿态赋给关键帧
    SetPose(F.mTcw);
}

/// 计算词袋表示
void KeyFrame::ComputeBoW() {
    /// 如果词袋向量或者特征点向量为空,执行
    if (mBowVec.empty() || mFeatVec.empty()) {
        /// 将描述子转化为描述子向量
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        /// 将描述子转化为词袋模型
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

/// 给当前帧位姿赋值
void KeyFrame::SetPose(const cv::Mat& Tcw_) {
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc * tcw;

    /// 计算当前位姿的逆
    Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc.rowRange(0, 3).col(3));

    /// center为相机坐标系（左目）中立体相机中心的坐标
    /// 立体相机中心点坐标与左目相机坐标之间只是在x轴上相差mHalfBaseline,
    /// 立体相机中两个摄像头的连线为x轴，正方向为左目相机指向右目相机 (齐次坐标)
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    /// 世界坐标系下，左目相机中心到立体相机中心的向量，方向由左目相机指向立体相机中心
    Cw = Twc * center;
}

/// 获取位姿
cv::Mat KeyFrame::GetPose() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

/// 获取位姿的逆
cv::Mat KeyFrame::GetPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

/// 获取(左目)相机的中心在世界坐标系的位置
cv::Mat KeyFrame::GetCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

/// 获取双目相机的中心,这个只有在可视化的时候才会用到
cv::Mat KeyFrame::GetStereoCenter() {
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

/// 获取旋转,世界坐标系到相机坐标系
cv::Mat KeyFrame::GetRotation() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

/// 获取平移,世界坐标系到相机坐标系
cv::Mat KeyFrame::GetTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).col(3).clone();
}

/**
 * 添加连接
 * @param pKF       需要关联的关键帧
 * @param weight    权重,当前关键帧与pKF共同观测的路标点数量
 */
void KeyFrame::AddConnection(KeyFrame* pKF, const int& weight) {
    {
        unique_lock<mutex> lock(mMutexConnections);
        /// std::map::count函数只可能返回0或1两种情况
        if(!mConnectedKeyFrameWeights.count(pKF))           /// 无连接则建立连接,用权重赋值
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight)  /// 有连接,当权重不同,重新赋值
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    /// 更新最佳共视
    UpdateBestCovisibles();
}

/**
 * @brief 更新最佳共视
 *         每个关键帧都用一个容器来记录与其他关键帧的weight,当发生关键帧添加,删除或权重变化时,
 *         需要对容器中weight进行重新排序,存储到mvpOrderedConnectedKeyFrames和mvOrderedWeights
 */
void KeyFrame::UpdateBestCovisibles() {
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, KeyFrame*>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    /// 使用pair存储原来map中的数据,以权重为first量,便于排序
    for (map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                 mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
       vPairs.push_back(make_pair(mit->second, mit->first));

    /// 默认按照first量:weight进行排序
    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;   // keyframe-list
    list<int> lWs;          // weight-list
    /// 链表push_front方式插入后,权重变为从大到小顺序
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    /// 权重从大到小
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

/// 获取与该关键帧连接的关键帧集合(没有顺序)
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames() {
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin();
        mit != mConnectedKeyFrameWeights.end(); mit++)
        s.insert(mit->first);
    return s;
}

/// 获取与该关键帧连接的关键帧(权重从大到小排序)
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() {
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/// 获取与该关键帧连接的前N个关键帧(已按权值排序)
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int& N) {
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        /// 如果数目不足N,就全部返回
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                 mvpOrderedConnectedKeyFrames.begin() + N);
}

/// 获取与该关键帧连接的权重大于等于(upper_bound，逆序中返回大于等于欧)w的关键帧
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int& w) {
    unique_lock<mutex> lock(mMutexConnections);
    /// 如果没有就返回空容器
    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),
                                            mvOrderedWeights.end(),
                                            w, KeyFrame::weightComp);
    /// 如果没有找到(最大的权重也比给定的阈值小)
    if (it == mvOrderedWeights.end() && *mvOrderedWeights.rbegin() < w) // 此处是否应该为.begin() < w
        return vector<KeyFrame*>();
    else {
        int n = it - mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                 mvpOrderedConnectedKeyFrames.begin() + n);
    }
}

/// 得到该关键帧与pKF的权重
int KeyFrame::GetWeight(KeyFrame* pKF) {
    unique_lock<mutex> lock(mMutexConnections);

    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        /// 没有连接的话权重也就是共视点个数就是0
        return 0;
}

/// 向关键帧中特征点索引中添加地图点
void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t& idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

/// 由于某些原因,导致当前关键帧观测到某个地图点被删除(bad==true),设置当前关键帧对应的地图点已经被删除了
void KeyFrame::EraseMapPointMatch(const size_t& idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
}

/// 同上
void KeyFrame::EraseMapPointMatch(MapPoint* pMP) {
    /// 其实和上面函数的操作差不多,不过是先从指针获取到索引,然后再进行删除罢了
    int idx = pMP->GetIndexInKeyFrame(this);
    if (idx >= 0)
        mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
}

/// 地图点的替换
void KeyFrame::ReplaceMapPointMatch(const size_t& idx, MapPoint* pMP) {
    mvpMapPoints[idx] = pMP;
}

/// 获取当前关键帧中的所有地图点
set<MapPoint*> KeyFrame::GetMapPoints() {
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
        /// 判断是否被删除了
        if (!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        /// 如果存在但标识为bad的也不返回
        if (!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/// 获取被观测相机数大于等于minObs的MapPoint
int KeyFrame::TrackedMapPoints(const int& minObs) {
    unique_lock<mutex> lock(mMutexFeatures);
    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    /// N是当前帧中特征点的个数
    for (int i = 0; i < N; i++) {
        MapPoint* pMP = mvpMapPoints[i];
        if (pMP) {                   /// 没有被删除
            if (!pMP->isBad()) {     /// 并且不是坏点
                if (bCheckObs) {
                    /// 该MapPoint是一个高质量的MapPoint
                    if (mvpMapPoints[i]->Observations() >= minObs)
                        nPoints++;
                } else /// 如果minObs==0 一定满足
                    nPoints++;
            }
        }
    }
    return nPoints;
}

/// 获取当前关键帧对应的地图点
vector<MapPoint*> KeyFrame::GetMapPointMatches() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

/// 获取当前关键帧中某个地图点
MapPoint* KeyFrame::GetMapPoint(const size_t& idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * 更新连接
 *    a.首先获得该关键帧关联的所有MapPoint点，然后遍历观测到这些点的其它所有关键帧,对每一个找到的关键帧，先存储到相应的容器
 *    b.计算所有共视帧与该帧的连接权重，权重即为共视的3d点的数量，连接按照权重从大到小进行排序,
 *      当该权重必须大于一个阈值，便在两帧之间建立边，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 *    c.更新covisibility graph，即把计算的边用来给图赋值，然后设置spanning tree中该帧的父节点，即共视程度最高的那一帧
 */
void KeyFrame::UpdateConnections() {
    map<KeyFrame*, int> KFcounter;  // 关键帧-权重，权重为其它关键帧与当前关键帧共视3d点的个数
    vector<MapPoint*> vpMP;
    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        /// 获得该关键帧关联的所有3D点
        vpMP = mvpMapPoints;
    }

    /// 统计每一个关键帧都有多少关键帧与它共视,统计结果放在KFcounter
    for (vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end();
         vit != vend; vit++) {
        MapPoint* pMP = *vit;
        if (!pMP)
            continue;

        if (pMP->isBad())
            continue;

        /// 对于每一个MapPoint点，observations记录了可以观测到该MapPoint的所有关键帧和对应特征索引
        map<KeyFrame*, size_t> observations = pMP->GetObservations();
        for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                     mend = observations.end(); mit != mend; mit++) {
            /// 除去自身，自己与自己不算共视
            if (mit->first->mnId == mnId)
                continue;
            /// 当前关键帧和其他关键帧各有多少个共视点
            KFcounter[mit->first]++;
        }
    }

    /// 一般不可能发生,但为了严谨
    if (KFcounter.empty())
        return;

    /// 如果共视数超过阈值则建立连接,如果都不超过阈值,则只保留最大的那个建立边
    int nmax = 0;
    KeyFrame* pKFmax = nullptr;
    int th = 15;

    vector<pair<int, KeyFrame*> > vPairs; // pair记录了与其他关键帧共视的次数
    vPairs.reserve(KFcounter.size());
    /// 对于一个和当前关键帧具有共视关系的关键帧
    for (map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end();
         mit != mend; mit++) {
        /// 更新具有最佳共视关系的关键帧信息
        if (mit->second > nmax) {
            nmax = mit->second;
            /// 找到对应权重最大的关键帧（共视程度最高的关键帧）
            pKFmax = mit->first;
        }
        /// 对应权重需要大于阈值，对这些关键帧建立连接
        if (mit->second >= th) {
            vPairs.push_back(make_pair(mit->second, mit->first));
            /// 为连接的关键帧进行权重更新
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    /// 如果没有超过阈值的权重，则对权重最大的关键帧建立连接
    if (vPairs.empty()) {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this, nmax);
    }

    /// vPairs里存的都是相互共视程度比较高的关键帧和共视权重, 按从小到大排序
    sort(vPairs.begin(), vPairs.end());
    /// 使用链表进行头部插入,变成从大到小排序
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (size_t i = 0; i < vPairs.size(); i++) {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);
        /// 更新(帧,权重)的连接图(数据结构为map)
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        /// 更新帧间的父子层级关系
        if (mbFirstConnection && mnId != 0) {
            /// 初始化该关键帧的父关键帧为共视程度最高的那个关键帧
            mpParent = mvpOrderedConnectedKeyFrames.front();
            /// 建立双向连接关系
            mpParent->AddChild(this);
            mbFirstConnection = false; /// 如此已经不是初始化树的状态了
        }
    }
}

/// 添加子关键帧（即和子关键帧具有最大共视关系的关键帧就是当前关键帧）
void KeyFrame::AddChild(KeyFrame* pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

/// 删除某个子关键帧
void KeyFrame::EraseChild(KeyFrame* pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

/// 改变当前关键帧的父关键帧
void KeyFrame::ChangeParent(KeyFrame* pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    // 添加双向连接关系
    mpParent = pKF;
    pKF->AddChild(this);
}

/// 获取当前关键帧的子关键帧
set<KeyFrame*> KeyFrame::GetChilds() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

/// 获取当前关键帧的父关键帧
KeyFrame* KeyFrame::GetParent() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

/// 判断某个关键帧是否是当前关键帧的子关键帧
bool KeyFrame::hasChild(KeyFrame* pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}


/// 给当前关键帧添加回环边，回环边连接了形成闭环关系的关键帧
void KeyFrame::AddLoopEdge(KeyFrame* pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

/// 获取和当前关键帧形成闭环关系的关键帧
set<KeyFrame*> KeyFrame::GetLoopEdges() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

/// 设置当前关键帧不要在优化的过程中被删除. 由回环检测线程调用
void KeyFrame::SetNotErase() {
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

/// 删除当前的这个关键帧,表示不进行回环检测过程,由回环检测线程调用
void KeyFrame::SetErase() {
    {
        unique_lock<mutex> lock(mMutexConnections);
        /// 如果当前关键帧和其他的关键帧没有形成回环关系,直接删除
        if (mspLoopEdges.empty()) {
            mbNotErase = false;
        }
    }

    /// SetBadFlag函数就是将mbToBeErased置为true，mbToBeErased就表示该KeyFrame被擦除了
    if (mbToBeErased) {
        SetBadFlag();
    }
}

/// 删除与该帧相关的所有连接关系,这里有个关键问题，就是该关键帧可能是其他节点的父节点，在删除之前需要给子节点换个爸爸
void KeyFrame::SetBadFlag() {
    /// 首先处理一下不能删除的关键帧的特殊情况
    {
        unique_lock<mutex> lock(mMutexConnections);
        /// 第0关键帧不允许被删除
        if (mnId == 0)
            return;
        /// mbNotErase表示不该擦除该KeyFrame，于是把mbToBeErased置为true，表示已经擦除了(其实没有擦除)
        else if (mbNotErase) {
            mbToBeErased = true;
            return;
        }
    }
    /// 删除可以删除的,让其它的KeyFrame删除与自己的联系
    for (map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                 mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        mit->first->EraseConnection(this);
    /// 删除与自己有关联的所有MapPoint
    for (size_t i = 0; i < mvpMapPoints.size(); i++)
        if (mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);

    /// 然后对当前关键帧成员变量的操作
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);
        /// 清空自己与其它关键帧之间的联系
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        /// 给子关键帧选择父关键帧
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        /// 如果这个关键帧有自己的孩子关键帧，给这些子关键帧赋予新的父关键帧
        while (!mspChildrens.empty()) {
            bool bContinue = false;
            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;
            /// 遍历每一个子关键帧，让它们更新它们指向的父关键帧
            for (set<KeyFrame*>::iterator sit = mspChildrens.begin(),
                         send = mspChildrens.end(); sit != send; sit++) {
                KeyFrame* pKF = *sit;
                /// 跳过bad子关键帧
                if (pKF->isBad())
                    continue;
                /// 子关键帧遍历每一个与它相连的关键帧（共视关键帧）
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
                    for (set<KeyFrame*>::iterator spcit = sParentCandidates.begin(),
                                 spcend = sParentCandidates.end(); spcit != spcend; spcit++) {
                        /// 如果该帧的子节点和父节点（祖孙节点）之间存在连接关系（共视）
                        /// 举例：B-->A（B的父节点是A） C-->B（C的父节点是B） D--C（D与C相连） E--C（E与C相连） F--C（F与C相连） D-->A（D的父节点是A） E-->A（E的父节点是A）
                        ///      现在B挂了，于是C在与自己相连的D、E、F节点中找到父节点指向A的D
                        ///      此过程就是为了找到可以替换B的那个节点。
                        /// 上面例子中，B为当前要设置为SetBadFlag的关键帧
                        ///           A为spcit，也即sParentCandidates
                        ///           C为pKF,pC，也即mspChildrens中的一个
                        ///           D、E、F为vpConnected中的变量，由于C与D间的权重 比 C与E间的权重大，因此D为pP
                        if (vpConnected[i]->mnId == (*spcit)->mnId) {
                            int w = pKF->GetWeight(vpConnected[i]);
                            /// 寻找并更新权值最大的那个共视关系
                            if (w > max) {
                                pC = pKF;                   // 子关键帧
                                pP = vpConnected[i];        // 目前和子关键帧具有最大权值的关键帧
                                max = w;                    // 这个最大的权值
                                bContinue = true;           // 说明子节点找到了可以作为其新父关键帧的帧
                            }
                        }
                    }
                }
            }
            /// 如果在上面的过程中找到了新的关键帧, 原来该帧的子节点都升级为父节点
            if (bContinue) {
                /// 因为父节点死了，并且子节点找到了新的父节点，子节点更新自己的父节点
                pC->ChangeParent(pP);
                /// 因为子节点找到了新的父节点并更新了父节点，那么该子节点升级，作为其它子节点的备选父节点
                sParentCandidates.insert(pC);
                /// 该子节点处理完毕
                mspChildrens.erase(pC);
            }
            else
                break;
        }
        /// 如果还有子节点没有找到新的父节点,设置当前帧的父关键帧
        if (!mspChildrens.empty()) {
            for (set<KeyFrame*>::iterator sit = mspChildrens.begin();
                 sit != mspChildrens.end(); sit++) {
                /// 直接把父节点的父节点作为自己的父节点 即对于这些子节点来说,他们的新的父节点其实就是自己的爷爷节点
                (*sit)->ChangeParent(mpParent);
            }
        }

        /// 当前帧的父节点中删除自己
        mpParent->EraseChild(this);
        /// 如果当前关键帧要被删除的话就要计算这个,表示当前关键帧到原本的父关键帧的位姿变换
        /// 注意在这个删除的过程中,其实并没有将当前关键帧中存储的父关键帧的指针删除掉
        mTcp = Tcw * mpParent->GetPoseInverse(); /// mTcp parent2child
        /// 到此为止，该关键帧被处理完毕
        mbBad = true;
    }
    /// 地图中删除当前帧
    mpMap->EraseKeyFrame(this);
    /// 关键帧数据库中删除当前帧
    mpKeyFrameDB->erase(this);
}

/// 返回当前关键帧是否bad
bool KeyFrame::isBad() {
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

/// 删除当前关键帧和指定关键帧之间的共视关系
void KeyFrame::EraseConnection(KeyFrame* pKF) {
    /// 其实这个应该表示是否真的是有共视关系
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF)) {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }
    /// 如果是真的有共视关系,发生了删除才更新共视关系,这样可以提高效率
    if(bUpdate)
        UpdateBestCovisibles();
}

/// 获取某个特征点邻域中的特征点id,和Frame.cc中对应函数相似; r为边长（半径）
vector<size_t> KeyFrame::GetFeaturesInArea(const float& x,
                                           const float& y,
                                           const float& r) const {
    vector<size_t> vIndices;
    vIndices.reserve(N);

    /// 列边界
    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;
    const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;
    /// 行边界
    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;
    const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    /// 遍历每个cell,取出其中每个cell中的点,并且每个点都要计算是否在邻域内
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for(int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            const vector<size_t> vCell = mGrid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;
                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}

/// 判断某个点是否在当前关键帧的图像中
bool KeyFrame::IsInImage(const float& x, const float& y) const {
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

/// 在双目和RGBD情况下将特征点反投影到空间中
cv::Mat KeyFrame::UnprojectStereo(int i) {
    const float z = mvDepth[i];
    if (z > 0) {
        /// 由2维图像反投影到相机坐标
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
    } else
        return cv::Mat();
}

/// 评估当前关键帧场景深度，q=2表示中值. 只是在单目情况下才会使用
float KeyFrame::ComputeSceneMedianDepth(const int q) {
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2, 3);
    /// 遍历每一个地图点,计算并保存其在当前关键帧下的深度
    /// 依据lambda*x(u, v, 1) = P*y, lambda=P的第3行和y的点乘
    for (int i = 0; i < N; i++) {
        if (mvpMapPoints[i]) {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw; // (R*x3Dw+t)的第三行，即z
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());
    return vDepths[(vDepths.size() - 1) / q];
}

} //namespace ORB_SLAM
