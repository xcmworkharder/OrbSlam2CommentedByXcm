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
#include "Map.h"

namespace ORB_SLAM2 {

///构造函数,地图点中最大关键帧id初始化为0
Map::Map() : mnMaxKFid(0) {
}

/// 在地图中插入关键帧,同时更新关键帧的最大id
void Map::AddKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if (pKF->mnId > mnMaxKFid)
        mnMaxKFid = pKF->mnId;
}

/// 向地图中插入地图点
void Map::AddMapPoint(MapPoint* pMP) {
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

/// 从地图容器中清除地图点,这个地图点所占用的内存空间并没有被释放
void Map::EraseMapPoint(MapPoint* pMP) {
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
}

/// 从第图关键帧容器中清除该关键帧
void Map::EraseKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

/// 设置参考地图点用于绘图显示局部地图点（红色）
void Map::SetReferenceMapPoints(const vector<MapPoint*>& vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

/// 这个好像没有用到
void Map::InformNewBigChange() {
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

/// 获取最大的变化id,也是当前在程序中没有被被用到过
int Map::GetLastBigChangeIdx() {
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

/// 获取地图中的所有关键帧
vector<KeyFrame*> Map::GetAllKeyFrames() {
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

/// 获取地图中的所有地图点
vector<MapPoint*> Map::GetAllMapPoints() {
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}

/// 获取地图点数目
long unsigned int Map::MapPointsInMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

/// 获取地图中的关键帧数目
long unsigned int Map::KeyFramesInMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

/// 获取参考地图点
vector<MapPoint*> Map::GetReferenceMapPoints() {
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

/// 获取地图中最大的关键帧id
long unsigned int Map::GetMaxKFid() {
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

/// 清空地图中的数据, 清除了容器空间和对应元素内存
void Map::clear() {
    for (set<MapPoint*>::iterator sit = mspMapPoints.begin(),
                 send = mspMapPoints.end(); sit != send; sit++)
        delete *sit;

    for (set<KeyFrame*>::iterator sit = mspKeyFrames.begin(),
                 send = mspKeyFrames.end(); sit != send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
