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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include <opencv2/core/core.hpp>
#include <mutex>


namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;

/**
 * @brief MapPoint地图中的特征点,包括三维坐标和描述子
 * 1. 维护关键帧的共视关系
 * 2. 描述计算向量之间的距离,在多个关键帧的特征点中找最匹配的特征点
 * 3. 闭环完成修正后,需要根据修正的主帧位姿修正特征点
 * 4. 对于非关键帧,也产生MapPoint,给Tracking功能临时使用
 */
class MapPoint
{
public:
    /**
     * 关键帧地图点的构造函数
     * @param Pos       世界坐标系的位姿
     * @param pRefKF    关键帧
     * @param pMap      地图指针
     */
    MapPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap);

    /**
     * 普通帧地图的构造函数
     * @param Pos       世界坐标系的位姿
     * @param pMap      地图指针
     * @param pFrame    普通帧
     * @param idxF      MapPoint在普通帧中的索引,即对应特征点的编号
     */
    MapPoint(const cv::Mat& Pos, Map* pMap, Frame* pFrame, const int& idxF);

    /// 设置地图点世界坐标系的位姿
    void SetWorldPos(const cv::Mat& Pos);
    /// 获取地图点世界坐标系的位姿
    cv::Mat GetWorldPos();

    /// 获取地图点的平均观测方向
    cv::Mat GetNormal();
    /// 获取当前地图点的参考关键帧
    KeyFrame* GetReferenceKeyFrame();

    /// 获取观测到当前点的所有关键帧
    std::map<KeyFrame*, size_t> GetObservations();
    
    /// 获取当前地图点的被观测次数
    int Observations();

    /**
     * 添加观测信息,同时计算观测的相机数量,单目+1,双目+2
     * @param pKF 关键帧指针
     * @param idx 地图点在关键帧中对应特征点的索引
     */
    void AddObservation(KeyFrame* pKF, size_t idx);

    /**
     * 删除观测关系 删除对应索引,相机数量,指针等
     * @param pKF  关键帧指针
     */
    void EraseObservation(KeyFrame* pKF);

    /// 获取地图点在某个关键帧中对应的特征点ID, 没有置为-1
    int GetIndexInKeyFrame(KeyFrame* pKF);

    /// 判断地图点是否在关键帧中
    bool IsInKeyFrame(KeyFrame* pKF);

    /// 删除地图点，并清除关键帧和地图中所有和该地图点对应的关联关系
    void SetBadFlag();

    /// 判断该地图点是否为Bad
    bool isBad();

    /// 替换地图点, 将this地图点替换为pMP,因为使用闭环后,需要重新建立地图点和关键帧的关系
    void Replace(MapPoint* pMP);
    /// 获取替换的地图点
    MapPoint* GetReplaced();

    /// 增加地图点被图像帧可视的次数, 可视不一定被找到被匹配
    void IncreaseVisible(int n = 1);
    /// 增加地图点被图像帧中特征点匹配的次数
    void IncreaseFound(int n = 1);
    /// 返回被找到匹配次数占可视的比例
    float GetFoundRatio();
    /// 返回被找到匹配的次数
    inline int GetFound() const {
        return mnFound;
    }

    /// 计算最匹配的描述子
    void ComputeDistinctiveDescriptors();

    /// 获取当前地图点的描述子
    cv::Mat GetDescriptor();

    /// 更新法向量和深度值
    void UpdateNormalAndDepth();

    /// 获取尺度不变的最小距离 ToConfirm
    float GetMinDistanceInvariance();
    /// 获取尺度不变的最大距离 ToConfirm
    float GetMaxDistanceInvariance();
    /// 预测尺度(关键帧)
    int PredictScale(const float& currentDist, KeyFrame* pKF);
    /// 预测尺度(普通帧)
    int PredictScale(const float& currentDist, Frame* pF);

public:
    long unsigned int mnId;             // 当前地图点的全局编号
    static long unsigned int nNextId;   // 用于记录并确保新创建地图点统一编号
    const long int mnFirstKFid;         // 创建该MapPoint的关键帧ID
    const long int mnFirstFrame;        // 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）
    int nObs;                           // 被观测到的次数

    /// 跟踪使用的变量
    float mTrackProjX;                  // 当前地图点投影到某帧上后的坐标
    float mTrackProjY;                  // 当前地图点投影到某帧上后的坐标
    float mTrackProjXR;                 // 当前地图点投影到某帧上后的坐标(右目)
    int mnTrackScaleLevel;              // 所处的尺度, 由其他的类进行操作 //?
    float mTrackViewCos;                // 被追踪到时,那帧相机看到当前地图点的视角

    /// 判断该点是否投影的布尔变量
    bool mbTrackInView;
    /// 防止将MapPoints重复添加至mvpLocalMapPoints的标记
    long unsigned int mnTrackReferenceForFrame;

    /// 决定是否进行isInFrustum判断的变量
    long unsigned int mnLastFrameSeen;

    /// ToConfirm
    long unsigned int mnBALocalForKF;          
    long unsigned int mnFuseCandidateForKF;

    /// 标记当前地图点是作为哪个"当前关键帧"的回环地图点(即回环关键帧上的地图点),在回环检测线程中被调用
    long unsigned int mnLoopPointForKF;
    /// 如果这个地图点对应的关键帧参与到了回环检测的过程中,那么在回环检测过程中已经使用了这个关键帧修正只有的位姿来修正了这个地图点,那么这个标志位置位
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    /// 全局BA优化后(如果当前地图点参加了的话),这里记录优化后的位姿
    cv::Mat mPosGBA;
    /// 如果当前点的位姿参与到了全局BA优化,那么这个变量记录了那个引起全局BA的"当前关键帧"的id
    long unsigned int mnBAGlobalForKF;

    ///全局BA中对当前点进行操作的时候使用的互斥量
    static std::mutex mGlobalMutex;

protected:
    /// 地图点在世界坐标系的坐标
    cv::Mat mWorldPos;

    /// 观测到该MapPoint的KF和该MapPoint在KF中的索引
    std::map<KeyFrame*, size_t> mObservations;

    /// 平均观测方向向量
    cv::Mat mNormalVector;

    /// 通过ComputeDistinctiveDescriptors() 得到的最优描述子
    cv::Mat mDescriptor;

    /// 参考关键帧, 通常就是创建该MapPoint的那个关键帧
    KeyFrame* mpRefKF;

    /// 被观测到和被匹配找到的次数
    int mnVisible;
    int mnFound;

    /// 设置地图点不可用待删除的标志
    bool mbBad;

    /// 当前地图点的点替换点(回环之后)
    MapPoint* mpReplaced;

    /// 尺度变化的最大和最小距离
    float mfMinDistance;
    float mfMaxDistance;

    /// 所属的地图
    Map* mpMap;

    /// 对当前地图点位姿进行操作的时候的互斥量
    std::mutex mMutexPos;
    /// 对当前地图点的特征信息进行操作的时候的互斥量
    std::mutex mMutexFeatures;
};
} //namespace ORB_SLAM

#endif // MAPPOINT_H
