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
#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
/// ORB-SLAM2系统中的主要功能模块
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2 {
/// 所用功能类的前置声明
class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

/**
* 主线程
* 其他各个模块都是由这里开始被调用的
*/
class System {
public:
    /// 这个枚举类型用于列举使用的传感器类型
    enum eSensor {
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:
    /// 构造函数，用来初始化整个系统
    System(const string& strVocFile,                    //指定ORB字典文件的路径
           const string& strSettingsFile,               //指定配置文件的路径
           const eSensor sensor,                        //指定所使用的传感器类型
           const bool bUseViewer = true);               //指定是否使用可视化界面

    /// 下面是针对三种不同类型传感器所设计的三种运动追踪接口,彩色图像为CV_8UC3类型，都会被转换成为灰度图像
    /// 追踪接口返回估计的相机位姿，如果追踪失败则返回NULL
    /// 注意这里双目图像有同步和校准的概念。
    cv::Mat TrackStereo(const cv::Mat& imLeft,          //左目图像
                        const cv::Mat& imRight,         //右目图像
                        const double& timestamp);       //时间戳

    /// 注意这里对RGBD图像的说法则是“配准”, registration
    cv::Mat TrackRGBD(const cv::Mat& im,                //彩色图像
                      const cv::Mat& depthmap,          //深度图像
                      const double& timestamp);         //时间戳

    cv::Mat TrackMonocular(const cv::Mat &im,           //图像
                           const double &timestamp);    //时间戳

    /// 停止局部建图线程,仅执行图像跟踪
    void ActivateLocalizationMode();
    /// 恢复局部建图线程,执行slam
    void DeactivateLocalizationMode();

    /// 判断从上次调用本函数后是否发生了比较大的地图变化(loop closure, global BA)
    bool MapChanged();

    /// 复位系统(清空地图)
    void Reset();

    /// 关闭系统,需要停止并等待所有线程结束,调用时机在保存轨迹之前
    void Shutdown();

    /// 以tum rgbd格式保存相机轨迹,不适用于单目情况,在系统Shutdown之前调用
    void SaveTrajectoryTUM(const string& filename);     //指定文件名

    /// 以TUM格式保存关键帧位姿,适用所有相机,在系统shutdown之前调用
    void SaveKeyFrameTrajectoryTUM(const string& filename);   //指定文件名

    /// 以KITTI格式保存相机的运行轨迹,适用stereo和rgbd格式,在系统shutdown之前调用
    void SaveTrajectoryKITTI(const string &filename);

    /// 在这里可以实现自己的地图保存和加载函数
    // SaveMap(const string& filename);
    // LoadMap(const string& filename);

    /// 获取最近的运动追踪状态、地图点追踪状态、特征点追踪状态, 在TrackMonocular (or stereo or RGBD)后调用
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

private:
    /// 变量命名方式，类变量有前缀m，指针类型还要多加个前缀p,进程变量加个前缀t
    /// 传感器类型
    eSensor mSensor;
    /// 一个指针指向ORB字典
    ORBVocabulary* mpVocabulary;
    /// 关键帧数据库的指针，这个数据库用于重定位和回环检测
    KeyFrameDatabase* mpKeyFrameDatabase;
    /// 指向地图（数据库）的指针,所有关键帧和地图点
    Map* mpMap;
    /// 图像追踪器指针，用于接收图像帧计算相机位姿,决定何时插入关键帧,跟踪失败时创建新地图点并重定位
    Tracking* mpTracker;
    /// 局部建图器指针, 管理局部地图,执行局部BA
    LocalMapping* mpLocalMapper;
    /// 回环检测器指针，使用关键帧搜寻回环,发现回环先执行位姿图方式的优化,然后在执行一个全ba
    LoopClosing* mpLoopCloser;
    /// 可视化器指针，使用pangolin绘制地图和相机位姿
    Viewer* mpViewer;
    /// 帧绘制器指针
    FrameDrawer* mpFrameDrawer;
    /// 地图绘制器指针
    MapDrawer* mpMapDrawer;
    /// 系统线程(跟踪线程在主线程中),其他包块局部建图, 回环 和 可视化线程
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;
    /// 复位标志
    std::mutex mMutexReset;
    bool mbReset;
    /// 模式改变标志
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;
    /// 追踪状态标志和追踪信息
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;
};

} // namespace ORB_SLAM

#endif // SYSTEM_H
