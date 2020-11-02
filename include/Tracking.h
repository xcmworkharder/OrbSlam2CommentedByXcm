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
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include <mutex>

namespace ORB_SLAM2
{
class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

/**
 * @brief 当前帧追踪类
 */
class Tracking {
public:
    /**
     * @brief 构造函数
     * @param[in] pSys              系统实例 
     * @param[in] pVoc              字典指针
     * @param[in] pFrameDrawer      帧绘制器
     * @param[in] pMapDrawer        地图绘制器
     * @param[in] pMap              地图指针
     * @param[in] pKFDB             关键帧数据库指针
     * @param[in] strSettingPath    配置文件路径
     * @param[in] sensor            传感器类型
     */
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer,
             MapDrawer* pMapDrawer, Map* pMap, KeyFrameDatabase* pKFDB,
             const string &strSettingPath, const int sensor);

    /// 进行数据预处理,调用跟踪函数,提取特征,执行双目匹配
    /// 下面的函数都是对不同的传感器输入的图像进行处理(转换成为灰度图像),并且调用Tracking线程
    /**
     * @brief 处理双目输入
     * @param[in] imRectLeft    左目图像
     * @param[in] imRectRight   右目图像
     * @param[in] timestamp     时间戳
     * @return cv::Mat          世界坐标系到该帧相机坐标系变换矩阵
     */
    cv::Mat GrabImageStereo(const cv::Mat& imRectLeft,
                            const cv::Mat& imRectRight,
                            const double& timestamp);

    /**
     * @brief 处理RGBD输入的图像
     * @param[in] imRGB         彩色图像
     * @param[in] imD           深度图像
     * @param[in] timestamp     时间戳
     * @return cv::Mat          世界坐标系到该帧相机坐标系变换矩阵
     */
    cv::Mat GrabImageRGBD(const cv::Mat& imRGB,const cv::Mat& imD,
                          const double& timestamp);

    /**
     * @brief 处理单目输入图像
     * @param[in] im            图像
     * @param[in] timestamp     时间戳
     * @return cv::Mat          世界坐标系到该帧相机坐标系的变换矩阵
     */
    cv::Mat GrabImageMonocular(const cv::Mat& im, const double& timestamp);

    /**
     * @brief 设置局部建图线程指针
     * @param[in] pLocalMapper 局部建图器
     */
    void SetLocalMapper(LocalMapping* pLocalMapper);

    /**
     * @brief 设置回环检测线程指针
     * @param[in] pLoopClosing 回环检测器
     */
    void SetLoopClosing(LoopClosing* pLoopClosing);

    /**
     * @brief 设置可视化查看线程指针
     * @param[in] pViewer 可视化查看器
     */
    void SetViewer(Viewer* pViewer);

    /**
     * @brief 导入新的配置信息,焦距要近似,否则尺度预测可能会失败
     * @param[in] strSettingPath 配置文件路径
     */
    void ChangeCalibration(const string& strSettingPath);

    /**
     * @brief 设置进入仅定位模式
     * @param[in] flag 设置仅仅进行跟踪的标志位
     */
    void InformOnlyTracking(const bool& flag);

public:
    /// 跟踪状态类型
    enum eTrackingState{
        SYSTEM_NOT_READY = -1,        //系统没有准备好状态,系统加载配置阶段状态
        NO_IMAGES_YET = 0,            //当前无图像
        NOT_INITIALIZED = 1,          //有图像但是没有完成初始化
        OK = 2,                       //正常时候的工作状态
        LOST = 3                      //系统已经跟丢了的状态
    };

    /// 当前跟踪状态
    eTrackingState mState;
    /// 上一帧的跟踪状态
    eTrackingState mLastProcessedState;
    /// 传感器类型:MONOCULAR, STEREO, RGBD
    int mSensor;

    /// 追踪线程的当前帧
    Frame mCurrentFrame;
    /// 在双目输入和在RGBD输入时，为左侧图像的灰度图
    cv::Mat mImGray;

    /// ----初始化变量-------------------------
    /// 初始化阶段的上一次匹配信息
    std::vector<int> mvIniLastMatches;
    /// 初始化阶段当前帧的匹配信息
    std::vector<int> mvIniMatches;
    /// 在初始化的过程中,保存参考帧中的特征点
    std::vector<cv::Point2f> mvbPrevMatched;
    /// 初始化过程中匹配后进行三角化得到的空间点
    std::vector<cv::Point3f> mvIniP3D;
    /// 初始化过程中的参考帧
    Frame mInitialFrame;
    /// 所有的参考关键帧的位姿链表
    list<cv::Mat> mlRelativeFramePoses;
    /// 参考关键帧链表
    list<KeyFrame*> mlpReferences;
    /// 所有帧的时间戳链表
    list<double> mlFrameTimes;
    /// 是否跟丢的标志链表
    list<bool> mlbLost;

    /// 标记当前系统是处于SLAM状态还是纯定位状态
    bool mbOnlyTracking;

    /// 系统复位操作
    void Reset();

protected:
    /// 主追踪函数,与传感器类型无关
    void Track();

    /// stereo和RGBD地图初始化
    void StereoInitialization();

    /// 单目地图初始化
    void MonocularInitialization();
   
    /// 产生单目输入的初始地图
    void CreateInitialMapMonocular();

    /// 检查上一帧的关键点是否能够被替换
    void CheckReplacedInLastFrame();

    /// 跟踪参考关键帧
    bool TrackReferenceKeyFrame();

    /// 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
    void UpdateLastFrame();

    /// 根据匀速度模型对上一帧的MapPoints进行跟踪
    bool TrackWithMotionModel();

    /// 系统重定位
    bool Relocalization();

    /// 更新局部地图 LocalMap
    void UpdateLocalMap();

    /// 更新局部地图点
    void UpdateLocalPoints();

    /// 更新局部关键帧
    void UpdateLocalKeyFrames();

    /// 对Local Map的MapPoints进行跟踪
    bool TrackLocalMap();

    /// 在局部地图中查找当前帧视野范围内的点
    void SearchLocalPoints();

    /// 判断是否需要新关键帧
    bool NeedNewKeyFrame();

    /// 创建新的关键帧
    void CreateNewKeyFrame();

    /// 当进行纯定位时才会有的一个变量,当变量为true表示地图无匹配点为
    bool mbVO;

    /// 局部建图器指针
    LocalMapping* mpLocalMapper;
    /// 回环检测器指针
    LoopClosing* mpLoopClosing;

    /// ORB特征点提取器
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    /// ORB特征字典
    ORBVocabulary* mpORBVocabulary;
    /// 当前系统运行的时候,关键帧所产生的数据库
    KeyFrameDatabase* mpKeyFrameDB;

    /// 单目初始器
    Initializer* mpInitializer;

    /// 参考关键帧
    KeyFrame* mpReferenceKF; // 当前关键帧就是参考帧
    /// 局部关键帧集合
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    /// 局部地图点的集合
    std::vector<MapPoint*> mvpLocalMapPoints;

    /// 指向系统实例的指针
    System* mpSystem;

    /// 查看器对象指针
    Viewer* mpViewer;
    /// 帧绘制器指针
    FrameDrawer* mpFrameDrawer;
    /// 地图绘制器指针
    MapDrawer* mpMapDrawer;

    /// (全局)地图指针
    Map* mpMap;

    /// 相机的内参数矩阵
    cv::Mat mK;
    /// 相机的去畸变参数
    cv::Mat mDistCoef;
    /// 相机的基线长度 * 相机的焦距
    float mbf;

    /// 新关键帧最大最小帧数(根据帧频)
    int mMinFrames;
    int mMaxFrames;

    /// 用于区分远点和近点的阈值. 近点认为可信度比较高;远点则要求在两个关键帧中得到匹配
    float mThDepth;

    /// 深度缩放因子,只对RGBD输入有效,对于TUM数据深度值被缩放
    float mDepthMapFactor;

    /// 当前帧中匹配的内点,将会被不同的函数反复使用
    int mnMatchesInliers;

    /// 上一关键帧
    KeyFrame* mpLastKeyFrame;
    /// 上一帧
    Frame mLastFrame;
    /// 上一个关键帧的ID
    unsigned int mnLastKeyFrameId;
    /// 上一次重定位帧的ID
    unsigned int mnLastRelocFrameId;

    /// 运动模型,当前是恒速模型
    cv::Mat mVelocity;

    /// RGB图像的颜色通道顺序, true:RGB false:BGR, grayscale时忽略
    bool mbRGB;

    /// 临时地图点,用于提高stereo和RGBD摄像头的帧间效果,用完之后就扔了
    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
