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
#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>

namespace ORB_SLAM2 {

/// 系统的构造函数，将会启动其他的线程
System::System(const string& strVocFile,					    // 词典文件路径
			   const string& strSettingsFile,				    // 配置文件路径
			   const eSensor sensor,						    // 传感器类型
               const bool bUseViewer):						    // 是否使用可视化界面
					 mSensor(sensor), 						    // 初始化传感器类型
					 mpViewer(static_cast<Viewer*>(nullptr)),   // 未创建可视化器
					 mbReset(false),						    // 默认系统未启动
					 mbActivateLocalizationMode(false),		    // 且启动跟踪定位模式
        			 mbDeactivateLocalizationMode(false) {	    // 未切换局部建图模式
    /// 显示版权信息
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    /// 输出当前使用传感器类型
    cout << "Input sensor was set to: ";
    if (mSensor == MONOCULAR)
        cout << "Monocular" << endl;
    else if (mSensor == STEREO)
        cout << "Stereo" << endl;
    else if (mSensor == RGBD)
        cout << "RGB-D" << endl;

    /// 定义配置读取文件
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    /// 如果打开失败，提示并退出程序
    if (!fsSettings.isOpened()) {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    /// 新建ORB字典类对象
    mpVocabulary = new ORBVocabulary();
    /// 获取字典加载状态
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    /// 如果加载失败，输出错信息并退出系统
    if (!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    /// 输出加载成功信息
    cout << "Vocabulary loaded!" << endl << endl;

    /// 创建关键帧数据库
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    /// 创建地图类对象
    mpMap = new Map();

    /// 创建图像帧绘制类对象,被可视化类使用
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    /// 在本主进程中初始化追踪线程,存在主线程中
    mpTracker = new Tracking(this,						//当前系统类指针
    						 mpVocabulary,				//字典
    						 mpFrameDrawer, 			//帧绘制器
    						 mpMapDrawer,				//地图绘制器
                             mpMap, 					//地图
                             mpKeyFrameDatabase, 		//关键帧地图
                             strSettingsFile, 			//设置文件路径
                             mSensor);					//传感器类型

    /// 局部建图线程执行类
    mpLocalMapper = new LocalMapping(mpMap, 				    //地图类指针
    								 mSensor == MONOCULAR);	    //是否使用单目相机
    /// 局部建图线程
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,	//这个线程会调用的函数
    							 mpLocalMapper);				//这个调用函数的参数

    /// 回环检测线程
    mpLoopCloser = new LoopClosing(mpMap, 						//地图
    							   mpKeyFrameDatabase, 			//关键帧数据库
    							   mpVocabulary, 				//ORB字典
    							   mSensor != MONOCULAR);	    //当前的传感器是否是单目
    /// 创建回环检测线程
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run,	//线程的主函数
    							mpLoopCloser);					//该函数的参数
    /// 创建可视化线程
    if (bUseViewer) {
    	/// 新建viewer
        mpViewer = new Viewer(this, 			//主线程中调用
        					  mpFrameDrawer,	//帧绘制器
        					  mpMapDrawer,		//地图绘制器
        					  mpTracker,		//追踪器
        					  strSettingsFile);	//配置文件的访问路径
        /// 新建viewer线程
        mptViewer = new thread(&Viewer::Run, mpViewer);
        /// 给运动追踪器设置其查看器
        mpTracker->SetViewer(mpViewer);
    }
    /// 设置进程间的指针
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

/// 双目输入时的追踪器接口
cv::Mat System::TrackStereo(const cv::Mat& imLeft, 		// 左侧图像
							const cv::Mat& imRight, 	// 右侧图像
							const double& timestamp) {	// 时间戳
	/// 检查输入数据类型是否合法
    if (mSensor != STEREO) {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }

    /// 运行模式的改变,使用代码段便于lock自动析构解除
    {
        unique_lock<mutex> lock(mMutexMode);
        /// 如果设置为激活定位模式
        if (mbActivateLocalizationMode) {
        	/// 调用局部建图器的请求停止函数
            mpLocalMapper->RequestStop();
            /// 等待局部建图线程结束
            while (!mpLocalMapper->IsStopped()) {
                usleep(1000);
            }
            /// 运行到这里的时候，局部建图线程应该已经地停止了,定位时只有追踪工作
            mpTracker->InformOnlyTracking(true);
            /// 同时清除定位激活标记,防止重复执行激活动作
            mbActivateLocalizationMode = false;//
        }

        /// 如果设置为取消定位模式
        if (mbDeactivateLocalizationMode) {
        	/// 告知追踪器，现在地图构建部分也要开始工作了
            mpTracker->InformOnlyTracking(false);
            /// 局部建图器要开始工作
            mpLocalMapper->Release();
            /// 清楚设置取消标志,防止重复执行
            mbDeactivateLocalizationMode = false;
        }
    }

    /// 系统复位操作
    {
	    unique_lock<mutex> lock(mMutexReset);
	    /// 是否有复位请求
	    if (mbReset) {
	    	/// 追踪器复位
	        mpTracker->Reset();
	        /// 清除标志
	        mbReset = false;
	    }
    }

    /// 运动追踪器GrabImageStereo函数计算矩阵相机位姿Tcw
    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

    /// 给运动追踪状态上锁
    unique_lock<mutex> lock2(mMutexState);
    /// 获取运动追踪状态
    mTrackingState = mpTracker->mState;
    /// 获取当前帧追踪到的地图点向量指针
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    /// 获取当前帧追踪到的关键帧特征点向量的指针
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    /// 返回获得的相机运动估计
    return Tcw;
}

/// RGBD跟踪器接口,相关功能与双目相机类型类似
cv::Mat System::TrackRGBD(const cv::Mat& im,
                          const cv::Mat& depthmap,
                          const double& timestamp) {
	/// 判断输入数据类型是否合法
    if (mSensor != RGBD) {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    /// 运行模式的改变,使用代码段便于lock自动析构解除
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();
            while (!mpLocalMapper->IsStopped()) {
                usleep(1000);
            }
            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }

        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    /// 系统复位操作
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    /// 获得相机位姿的估计
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

/// 单目追踪器接口,与单目,rgbd类似
cv::Mat System::TrackMonocular(const cv::Mat& im, const double& timestamp) {
    if(mSensor != MONOCULAR) {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }
    /// 运行模式的改变,使用代码段便于lock自动析构解除
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();
            while (!mpLocalMapper->IsStopped()) {
                usleep(1000);
            }
            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }

        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }
    /// 系统复位操作
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

/// 激活定位模式
void System::ActivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    ///设置标志
    mbActivateLocalizationMode = true;
}

/// 取消定位模式
void System::DeactivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

/// 判断是否地图有较大的改变
bool System::MapChanged() {
    static int n = 0;  // 全局索引
    /// 获得地图上一个变化的id
    int curn = mpMap->GetLastBigChangeIdx();
    if (n < curn) {
        /// 保存上一个变化的id
        n = curn;
        return true;
    } else
        return false;
}

/// 准备执行系统复位
void System::Reset() {
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

/// 系统退出
void System::Shutdown() {
	/// 对局部建图线程和回环检测线程发送终止请求
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    /// 终止可视化器
    if (mpViewer) {
    	/// 向查看器发送终止请求
        mpViewer->RequestFinish();
        /// 等到知道可视化器真正地停止
        while (!mpViewer->isFinished())
            usleep(5000);
    }

    /// 如果仍有未完成的线程,则继续等待
    while (!mpLocalMapper->IsFinished() || !mpLoopCloser->isFinished()  ||
    	   mpLoopCloser->isRunningGBA()) {
        usleep(5000);
    }

    if (mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

/// 以tum rgbd格式保存相机轨迹,不适用于单目情况,在系统Shutdown之前调用
void System::SaveTrajectoryTUM(const string& filename) {
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    /// 不适用于单目模式
    if (mSensor == MONOCULAR) {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    /// 从地图中获取所有关键帧
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    /// 根据关键帧mnId排序从小到大排序
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId/*函数指针*/);

    /// 获得从原点到世界系的转换,转换使得第一个关键帧在原点(因为经过回环之后它可能不在原点)
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    /// 输出文件定义
    ofstream f;
    f.open(filename.c_str());
    f << fixed; // 按照原始浮点数形式,防止进入科学计数法显示

    /// 帧位姿被存储为相对于参考关键帧的形式(通过BA和位姿图方式优化)
    /// 我们可以先获得关键帧位姿,在通过相对变换进行拼接得出帧位姿(没有跟踪到的帧位姿不保存)
    /// 对于每一帧,都有一个关键帧lRit, 时间戳lT和一个跟踪失败设置为true的标识(lbL)
    /// 关键帧列表的迭代器头
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    /// 时间戳列表的迭代器头
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    /// 每帧追踪状态列表的迭代器头
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    ///对于每一个mlRelativeFramePoses中的帧lit
    for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
        lend = mpTracker->mlRelativeFramePoses.end(); // lend提前获取能够提高一点效率
        lit != lend;
        lit++, lRit++, lT++, lbL++)	{
    	/// 如果该帧追踪失败，继续遍历
        if (*lbL)
            continue;
       	/// 获取其对应的参考关键帧
        KeyFrame* pKF = *lRit;

        /// 到参考系的变换矩阵初始化
        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        /// 如果参考关键帧已经被剔除,遍历扩展树找一个适合的关键帧
        while (pKF->isBad()) {
        	/// 更新关键帧变换矩阵的初始值
            Trw = Trw * pKF->mTcp;
            /// 查找关键帧的父关键帧
            pKF = pKF->GetParent();
        }
        /// 跳出循环后,pKF应该能够获取位姿
        /// 最终得到的是参考关键帧相对于世界坐标系的变换
        Trw = Trw * pKF->GetPose() * Two;

        /// 得出世界坐标系到当前帧的变换矩阵
        cv::Mat Tcw = (*lit) * Trw;
        /// 提取出旋转矩阵, 做了转置
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        /// 提取出平移向量
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        /// 用四元数表示旋转
        vector<float> q = Converter::toQuaternion(Rwc);

        /// 然后按照给定的格式输出到文件中
        f << setprecision(6) << *lT << " " <<  setprecision(9)
          << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    /// 关闭文件
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


/// 以TUM格式保存关键帧位姿,适用所有相机,在系统shutdown之前调用
void System::SaveKeyFrameTrajectoryTUM(const string& filename) {
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;
    /// 获取所有关键帧并按照mnId从小到大排序
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId/*函数指针*/);

    /// 注释了原点校正
    // cv::Mat Two = vpKFs[0]->GetPoseInverse();

    /// 输出文件设置
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    /// 对于每个关键帧
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        /// 这里注释了原点校正
        // pKF->SetPose(pKF->GetPose()*Two);
        /// 如果这个关键帧是bad那么就跳过
        if (pKF->isBad())
            continue;
        /// 提取关键帧的旋转变换,转化为四元数
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        /// 提取关键帧的平移变换,相机在世界坐标系的位置相当于t_wc
        cv::Mat t = pKF->GetCameraCenter();
        /// 按照给定的格式输出到文件中, tum文件使用" "分割
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " "
          << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    ///关闭文件
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

/// 以KITTI格式保存相机的运行轨迹,适用stereo和rgbd格式,在系统shutdown之前调用
/// 具体逻辑与SaveTrajectoryTUM类似,注释基本相同
void System::SaveTrajectoryKITTI(const string& filename) {
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    /// 检查输入数据类型合法性
    if (mSensor == MONOCULAR) {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }
    /// 获取所有关键帧并按照mnId从小到大排序
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
    /// 第一关键帧的原点校正
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                lend = mpTracker->mlRelativeFramePoses.end(); //
        lit != lend;lit++, lRit++, lT++) {
        ORB_SLAM2::KeyFrame* pKF = *lRit;
        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);
        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }
        Trw = Trw * pKF->GetPose() * Two;
        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);
        f << setprecision(9)
          << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1)  << " " << Rwc.at<float>(0, 2) << " "
          << twc.at<float>(0) << " "
          << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1)  << " " << Rwc.at<float>(1, 2) << " "
          << twc.at<float>(1) << " "
          << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1)  << " " << Rwc.at<float>(2, 2) << " "
          << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

/// 获取追踪器状态
int System::GetTrackingState() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

/// 获取追踪到的地图点,返回vector<MapPoint*>
vector<MapPoint*> System::GetTrackedMapPoints() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

/// 获取追踪到的关键帧的点
vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
