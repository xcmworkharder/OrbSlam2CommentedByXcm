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
#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include <iostream>
#include <cmath>
#include <mutex>

using namespace std;

namespace ORB_SLAM2 {
/// 构造函数
Tracking::Tracking(
    System *pSys,                       //系统实例
    ORBVocabulary* pVoc,                //BOW字典
    FrameDrawer *pFrameDrawer,          //帧绘制器指针
    MapDrawer *pMapDrawer,              //地图点绘制器指针
    Map *pMap,                          //地图指针
    KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库指针
    const string& strSettingPath,       //配置文件路径
    const int sensor):                  //传感器类型
        mState(NO_IMAGES_YET),                            //当前系统还没有准备好
        mSensor(sensor),                                
        mbOnlyTracking(false),                            //处于SLAM模式
        mbVO(false),                                      //处于纯跟踪模式，表示了当前跟踪状态好坏
        mpORBVocabulary(pVoc),          
        mpKeyFrameDB(pKFDB), 
        mpInitializer(static_cast<Initializer*>(nullptr)),//空指针
        mpSystem(pSys), 
        mpViewer(nullptr),                                //默认不用
        mpFrameDrawer(pFrameDrawer),
        mpMapDrawer(pMapDrawer), 
        mpMap(pMap), 
        mnLastRelocFrameId(0) {                           //默认为没有这个过程,设置为0,

    /// 从配置文件中加载相机参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    ///构造相机内参矩阵
    ///     |fx  0   cx|
    /// K = |0   fy  cy|
    ///     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    /// 图像矫正系数
    /// [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    /// 有些相机的畸变系数中会没有k3项
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    /// 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    /// 插入关键帧的最大和最小值
    mMinFrames = 0;
    mMaxFrames = fps;

    /// 输出参数信息
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    /// ORB参数
    /// 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    /// 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    /// 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    /// 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    /// 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    /// tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                                        fIniThFAST, fMinThFAST);

    /// 如果是双目tracking过程中,还会用用到mpORBextractorRight作为右目特征点提取器
    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    /// 单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD) {
        /// 设置判断一个3D点远/近的阈值 mbf * 35 / fx, 相当于基线长度的倍数
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD) {
        /// 深度相机的缩放因子
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
}

/// 设置局部建图器
void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

/// 设置回环检测器
void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
    mpLoopClosing = pLoopClosing;
}

/// 设置可视化查看器
void Tracking::SetViewer(Viewer* pViewer) {
    mpViewer = pViewer;
}

/**
 * Stereo图像跟踪提取函数,图像为(RGB、BGR、RGBA、GRAY)
 * @param imRectLeft    左侧图像
 * @param imRectRight   右侧图像
 * @param timestamp     时间戳
 * @return 世界坐标系到该帧相机坐标系的变换矩阵
 */
cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  const double& timestamp) {
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    /// 将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3) {
        if (mbRGB) {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        } else {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    } else if(mImGray.channels() == 4) { // 如果是4通道图像
        if (mbRGB) {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        } else {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }

    /// 构建并配置当前帧参数
    mCurrentFrame = Frame(
        mImGray,                //左目图像
        imGrayRight,            //右目图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //左目特征提取器
        mpORBextractorRight,    //右目特征提取器
        mpORBVocabulary,        //字典
        mK,                     //内参矩阵
        mDistCoef,              //去畸变参数
        mbf,                    //基线长度
        mThDepth);              //远点,近点的区分阈值

    /// 执行跟踪函数
    Track();
    /// 返回位姿
    return mCurrentFrame.mTcw.clone();
}

/**
 * RGBD图像跟踪提取函数
 * @param imRGB RGB图像
 * @param imD 深度图像
 * @param timestamp
 * @return 输出世界坐标系到到该帧帧相机坐标变换矩阵
 */
cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                const double& timestamp) {
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    /// 将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    } else if(mImGray.channels() == 4) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    /// 如果mDepthMapFactor明显大于1.0或者类型不为浮点型,进行转化操作
    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        /// 将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
        imDepth.convertTo(
            imDepth,            //输出图像
            CV_32F,             //输出图像的数据类型
            mDepthMapFactor);   //缩放系数

    /// 构建并配置当前帧参数
    mCurrentFrame = Frame(
        mImGray,                //灰度图像
        imDepth,                //深度图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //ORB特征提取器
        mpORBVocabulary,        //词典
        mK,                     //相机内参矩阵
        mDistCoef,              //相机的去畸变参数
        mbf,                    //相机基线*相机焦距
        mThDepth);              //内外点区分深度阈值

    /// 执行跟踪主函数
    Track();

    /// 返回世界坐标系到当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}

/**
 * 单目图像跟踪提取函数
 * @param im 单目图像
 * @param timestamp 时间戳
 * @return 输出世界坐标系到该帧相机坐标系的变换矩阵
 */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im,
                                     const double& timestamp) {
    mImGray = im;
    /// 将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    } else if (mImGray.channels() == 4) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    /// 构建并配置当前帧参数
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary,
            mK, mDistCoef, mbf, mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary,
            mK, mDistCoef, mbf, mThDepth);

    /// 执行跟踪主函数
    Track();

    /// 返回世界坐标系到当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief 跟踪主函数, 与相机类型无关
 * Tracking 使用线程处理:估计运动、跟踪局部地图
 */
void Tracking::Track() {
    /// 如果图像没有图像,则设置为未初始化
    if (mState == NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    /// 存储Tracking最新的状态
    mLastProcessedState = mState;

    /// 设置地图互斥量,地图不能发生变化
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    /// 进行初始化
    if (mState == NOT_INITIALIZED) {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            /// 双目和RGBD相机的初始化使用如下这个函数
            StereoInitialization();
        else
            /// 单目初始化使用下面这个函数
            MonocularInitialization();

        /// 更新帧绘制器中状态
        mpFrameDrawer->Update(this);

        /// 如果初始化没有成功,则不会OK,直接返回
        if (mState != OK)
            return;
    } else { /// 如果不是NOT_INITIALIZED状态,进行跟踪
        bool bOK; // 临时变量,表示函数的返回状态
        /// 如果跟踪丢失状态,则使用运动模型或重新定位
        if (!mbOnlyTracking) { /// 如果处于SLAM模式
            if (mState == OK) {
                /// 局部建图可能会改变一些跟踪点,这里进行检查替换
                CheckReplacedInLastFrame();
                /// 如果运动该模型为空,或者刚完成重定位,进行参考关键帧跟踪
                if (mVelocity.empty() ||
                        mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    /// 调用参考关键帧跟踪函数
                    bOK = TrackReferenceKeyFrame();
                } else { /// 根据恒速模型设定当前帧的初始位姿
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                        /// 如果恒速跟踪失败了,还是使用参考关键帧跟踪函数
                        bOK = TrackReferenceKeyFrame();
                }
            } else { /// 如果处于丢失状态,则进行重定位
                bOK = Relocalization();
            }
        } else {///如果处于仅定位模式
            /// 如果跟丢了, 进行重定位
            if (mState == LOST) {
                bOK = Relocalization();
            } else { /// 如果没有跟丢
                /// mbVO为mbOnlyTracking为true时才有效
                if (!mbVO) { /// mbVO为false表示此帧匹配了很多MapPoints，跟踪很正常，
                    if (!mVelocity.empty()) { /// 如果有恒速模型
                        bOK = TrackWithMotionModel();
                        /// 为确保周全,可增加如下代码
                        // if(!bOK)
                        //    bOK = TrackReferenceKeyFrame();
                    } else { /// 如果恒速模型满足,那么就只能够进行参考关键帧来定位
                        bOK = TrackReferenceKeyFrame();
                    }
                } else { /// mbVO为true表明此帧匹配了很少MapPoints,跟踪不正常
                    /// 在这种情况下,我们计算两个相机位姿一个来自运动该模型,一个来自重定位
                    /// 重定位成功就选择定位,否则使用VO方式
                    bool bOKMM = false;     // 运动该模型跟踪结果标识
                    bool bOKReloc = false;  // 重定位结果标识
                    
                    /// 运动模型中构造的地图点
                    vector<MapPoint*> vpMPsMM;
                    /// 追踪运动模型后发现的外点
                    vector<bool> vbOutMM;
                    /// 运动模型得到的位姿
                    cv::Mat TcwMM;

                    /// 当运动模型非空时,根据运动模型计算位姿
                    if (!mVelocity.empty()) {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    /// 使用重定位方法来得到当前帧位姿
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc) { /// 重定位没有成功，但跟踪成功的情况
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        /// 如果跟踪结果不好
                        if (mbVO) {
                            for (int i =0; i < mCurrentFrame.N; i++) {
                                /// 如果这个特征点形成了地图点,并且也不是外点
                                if (mCurrentFrame.mvpMapPoints[i]
                                    && !mCurrentFrame.mvbOutlier[i]) {
                                    /// 增加被观测的次数记录
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    } else if(bOKReloc) { /// 如果重定位成功
                        mbVO = false; /// 认为跟踪正常
                    }
                    /// 运行模型或重定位之一成功就返回跟踪成功标识
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        /// 更新当前帧的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        /// 如果有初始化的相机位姿估计和匹配点,跟踪更新局部地图
        if (!mbOnlyTracking) { /// 处于slam模式
            if(bOK)
                bOK = TrackLocalMap();
        } else { /// 处于vo模式
            /// 如果mbVO为true,则匹配点不好,需要重定位成功才进行局部地图更新
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        /// 根据上面的操作结果确定跟踪状态
        if (bOK)
            mState = OK;
        else
            mState = LOST;

        /// 更新帧绘图器
        mpFrameDrawer->Update(this);

        /// 如果跟踪良好,判断是否插入关键帧
        if (bOK) {
            /// 更新运动模型
            if (!mLastFrame.mTcw.empty()) {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().
                        copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().
                        copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            } else
                /// 否则生成空矩阵
                mVelocity = cv::Mat();

            /// 更新了地图绘制器
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            /// 清除UpdateLastFrame中为当前帧临时添加的MapPoints
            /// 清空VO匹配点
            for (int i = 0; i < mCurrentFrame.N; i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    /// 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
                    if (pMP->Observations() < 1) {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
                    }
            }

            /// 删除临时地图点,先释放链表中元素的内存
            for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),
                        lend = mlpTemporalPoints.end(); lit != lend; lit++) {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            /// 清空链表空间
            mlpTemporalPoints.clear();

            /// 判断是否需要添加关键帧
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            /// 删除那些在bundle adjustment中检测为outlier的3D map点(Huber function来判断)
            for (int i = 0; i < mCurrentFrame.N; i++) {
                /// 如果还存在这个地图点,且优化中判别为外点,则设置为nullptr(相当于剔除)
                if (mCurrentFrame.mvpMapPoints[i]
                   && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
            }
        }

        /// 如果初始化不就就跟丢了,进行系统复位
        if (mState == LOST) {
            /// 如果地图中关键帧信息过少的话,直接重置系统
            if (mpMap->KeyFramesInMap() <= 5) {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        /// 如果未来设置参考关键帧,则进行设置
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        /// 将当前帧更新为上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    /// 记录位姿信息，用于轨迹复现
    if (!mCurrentFrame.mTcw.empty()) {
        /// 关键帧存储的位姿,表示世界坐标系到参考关键帧的位姿(因此要取反)
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        /// 保存各种状态
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    } else {
        /// 如果跟踪失败，则相对位姿使用上一次帧的值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

/**
 * @brief 双目或rgbd初始化函数
 * 由于具有深度信息，直接生成MapPoints
 */
void Tracking::StereoInitialization() {
    /// 当前帧的特征点超过500进行初始化
    if (mCurrentFrame.N > 500) {
        /// 设定初始位姿到原点
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        /// 将当前帧初始化为关键帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        /// 在地图中添加该初始关键帧
        mpMap->AddKeyFrame(pKFini);

        /// 创建地图特征点和对应的关键帧
        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) { /// 具有正深度的点才会被构造地图点
                /// 通过反投影得到该特征点的3D坐标
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                /// 将3d点构建为地图特征点
                MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);

                /// 记录关键帧上观测到地图点的特征点索引
                pNewMP->AddObservation(pKFini, i);
                /// 计算地图点对应特征的描述子
                pNewMP->ComputeDistinctiveDescriptors();
                /// 更新地图点平均观测方向的几距离范围
                pNewMP->UpdateNormalAndDepth();

                /// 地图中添加该地图点
                mpMap->AddMapPoint(pNewMP);
                /// 关键帧特征点关联地图点
                pKFini->AddMapPoint(pNewMP, i);

                /// 为当前Frame的特征点与MapPoint之间建立索引
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }
        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;
        /// 在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        /// 当前帧更新为上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        /// 把当前（最新的）局部MapPoints作为ReferenceMapPoints,绘图时使用
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        /// 追踪成功标识
        mState = OK;
    }
}

/**
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization() {
    if (!mpInitializer) { /// 如果初始化器还没有被创建
        /// 单目初始帧的特征点数大于100才执行
        if (mCurrentFrame.mvKeys.size() > 100) {
            /// 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
            mInitialFrame = Frame(mCurrentFrame);
            /// 用当前帧更新上一帧
            mLastFrame = Frame(mCurrentFrame);
            /// 记录"上一帧"所有特征点
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            /// 先删除再创建
            if(mpInitializer)
                delete mpInitializer;
            /// 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame, 1.0, 200);
            /// 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }
    } else { /// 如果单目初始化器已经被创建
        /// 如果当前帧的特征点数太少,则重新构造初始化器
        if ((int)mCurrentFrame.mvKeys.size() <= 100) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(nullptr);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        /// 在mInitialFrame与mCurrentFrame中找匹配的特征点对
        ORBmatcher matcher(
            0.9,        //最佳的和次佳特征点评分比值阈值，这里比较宽松，跟踪时一般0.7
            true);      //检查特征点的方向
        int nmatches = matcher.SearchForInitialization(
            mInitialFrame, mCurrentFrame,    //初始化时的参考帧和当前帧
            mvbPrevMatched,                  //在初始化参考帧中提取得到的特征点
            mvIniMatches,                    //保存匹配关系
            100);                            //搜索窗口大小

        /// 如果初始化两帧之间的匹配点太少，重新初始化
        if (nmatches < 100) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(nullptr);
            return;
        }

        cv::Mat Rcw;                // Current Camera Rotation
        cv::Mat tcw;                // Current Camera Translation
        vector<bool> vbTriangulated;// Triangulated Correspondences

        /// 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if (mpInitializer->Initialize(
            mCurrentFrame,      //当前帧
            mvIniMatches,       //当前帧和参考帧的特征点的匹配关系
            Rcw, tcw,           //初始化得到的相机的位姿
            mvIniP3D,           //进行三角化得到的空间点集合
            vbTriangulated)) {  //记录哪些点被三角化了

            /// 初始化成功后，删除那些无法进行三角化的匹配点
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            /// 将初始化第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            /// 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            ///创建初始化地图点MapPoints,将3D点包装成MapPoint类型存入KeyFrame和Map
            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief 为单目摄像头三角化得到的点生成MapPoints
 */
void Tracking::CreateInitialMapMonocular() {
    /// 认为单目初始化时候的参考帧和当前帧都是关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);  // 第一帧
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);  // 第二帧

    /// 将初始关键帧,当前关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    /// 将关键帧插入到地图中
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    /// 创建地图点,并关联到关键帧
    for (size_t i = 0; i < mvIniMatches.size(); i++) {
        /// 没有匹配，跳过
        if (mvIniMatches[i] < 0)
            continue;

        /// 空间点的世界坐标
        cv::Mat worldPos(mvIniP3D[i]);

        /// 构造地图点
        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

        /// 关键帧把地图点添加到对应的特征点索引上
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        /// 设置MapPoint可以被哪个KeyFrame哪个特征点观测到
        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        /// 从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        /// 更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        /// 当前帧地图点信息设置
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        /// 在地图中添加该MapPoint
        mpMap->AddMapPoint(pMP);
    }

    /// 更新关键帧之间的关联关系,每个边有一个权重，是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    /// BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    /// 计算深度的中位信息
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;
    
    /// 如果不满足平均深度大于0,且当前帧中被观测到的地图点的数目大于100,重启系统
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    /// 将两帧之间的变换归一化到平均深度1的尺度下
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    /// 把3D点的尺度也归一化到1
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
        if (vpAllMapPoints[iMP]) {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    /// 插入关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    /// 单目初始化之后，得到的初始地图中的所有点都是局部地图点
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    /// 初始化成功
    mState = OK;
}

/**
 * 检查上一帧的关键点是否能够被替换
 * 如果存在替换点则逐个替换
 */
void Tracking::CheckReplacedInLastFrame() {
    /// 遍历上一帧的关键点
    for (int i = 0; i < mLastFrame.N; i++) {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        /// 如果该点,则判断是否有替换点,有则逐个替换
        if (pMP) {
            MapPoint* pRep = pMP->GetReplaced();
            if (pRep) {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 对参考关键帧中MapPoints进行跟踪
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame() {
    /// 将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    /// 首先对参考关键帧执行ORB匹配,如果有足够的匹配点,则执行pnp
    ORBmatcher matcher(0.7, true);
    vector<MapPoint*> vpMapPointMatches;

    /// 通过特征点BoW加快当前帧与参考帧之间的特征点匹配
    int nmatches = matcher.SearchByBoW(
        mpReferenceKF,          //参考关键帧
        mCurrentFrame,          //当前帧
        vpMapPointMatches);     //存储匹配关系

    /// 匹配数超过15才继续
    if(nmatches < 15)
        return false;

    /// 设置当前帧的初始位姿和匹配关系
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw); //用上一次的Tcw设置当前帧位姿可加速优化

    /// 通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    /// 剔除优化后的外点
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            /// 如果对应到的某个特征点是外点
            if (mCurrentFrame.mvbOutlier[i]) {
                /// 清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                /// 其实这里--也没有什么用了,因为后面也用不到它了
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                /// 匹配的内点计数++
                nmatchesMap++;
        }
    }
    return nmatchesMap >= 10;
}

/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些）
 * 可以通过深度值产生一些新的MapPoints,用来补充当前视野中的地图点数目,这些新补充的地图点就被称之为"临时地图点""
 */
void Tracking::UpdateLastFrame() {
    /// 根据参考关键帧更新位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;  //上一帧的参考KF
    /// ref_keyframe 到 lastframe的位姿
    cv::Mat Tlr = mlRelativeFramePoses.back();
    /// 计算世界坐标系到上一帧的变换: Tlr*Trw
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    /// 如果上一帧为关键帧，或者单目情况，则退出
    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR)
        return;

    /// 得到上一帧有深度值的特征点数据对(深度,对应id)
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);

    for (int i = 0; i < mLastFrame.N; i++) {
        float z = mLastFrame.mvDepth[i];
        if (z > 0) {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    /// 如果上一帧中没有有效深度的点,那么就直接退出了
    if (vDepthIdx.empty())
        return;

    /// 按照深度从小到大排序
    sort(vDepthIdx.begin(), vDepthIdx.end());

    /// 将距离比较近的点包装成MapPoints插入地图
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;
        bool bCreateNew = false;

        /// 如果上一帧中的没有该地图点,或者创建后就没有被观测到,那么就生成一个临时的地图点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1) {
            bCreateNew = true;
        }

        /// 如果需要创建新的临时地图点
        if (bCreateNew) {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
            /// 设置上一帧对应信息
            mLastFrame.mvpMapPoints[i] = pNewMP;
            /// 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        } else { /// 如果不需要创建新的 临时地图点
            nPoints++;
        }

        /// 当当前的点的深度已经超过了远点的阈值,并且已经这样处理了超过100个点的时候,说明就足够了
        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时） (因为传感器的原因，单目情况下仅仅凭借一帧没法生成可靠的地图点)
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配  NOTICE 加快了匹配的速度
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel() {
    /// 最小距离 < 0.9*次小距离 匹配成功
    ORBmatcher matcher(0.9, true);

    UpdateLastFrame();

    /// 根据Const Velocity Model估计当前帧的位姿
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    /// 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(nullptr));

    /// 设置匹配过程中的搜索半径
    int th;
    if (mSensor != System::STEREO)
        th = 15;
    else
        th = 7;

    /// 根据匀速度模型对上一帧MapPoints进行跟踪,根据上一帧特征点对应3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame,
                                              th, mSensor == System::MONOCULAR);
    /// 如果跟踪的点少，则扩大搜索半径再来一次
    if (nmatches < 20) {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint*>(nullptr));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,
                                              2 * th,mSensor == System::MONOCULAR);
    }
    /// 如果扩大搜索范围找到的匹配点还是不够20个,那么就认为运动跟踪失败了.
    if (nmatches < 20)
        return false;

    /// 使用所有匹配点进行位姿优化
    Optimizer::PoseOptimization(&mCurrentFrame);

    /// 优化位姿后剔除outlier的mvpMapPoints
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                /// 累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }    

    if (mbOnlyTracking) {
        /// 如果在纯定位过程中追踪的地图点非常少,那么这里的mbVO==true
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap>=10;
}

/**
 * @brief 对Local Map的MapPoints进行跟踪
 * Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
 * @return true         跟踪成功
 * @return false        跟踪失败
 */
bool Tracking::TrackLocalMap() {
    /// 更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();

    /// 在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
    SearchLocalPoints();

    /// 更新局部所有MapPoints后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    /// 更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            /// 由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            if (!mCurrentFrame.mvbOutlier[i]) {
                /// 找到该点的帧数mnFound 加 1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                ///查看当前是否是在纯定位过程
                if (!mbOnlyTracking) {
                    /// 该MapPoint被观测次数大于，就将mnMatchesInliers加1
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                }
                else
                    /// 记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                    mnMatchesInliers++;
            } else if (mSensor == System::STEREO)
                //如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
        }
    }

    /// 根据跟踪匹配数目及回环情况决定是否跟踪成功
    /// 如果最近进行了重定位,那么至少跟踪上了50个点才认为是跟踪上了
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames
        && mnMatchesInliers < 50)
        return false;

    /// 如果是正常的状态,只要跟踪的地图点大于30个就认为成功了
    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

/**
 * @brief 判断当前帧是否为关键帧
 * @return true if needed
 */
bool Tracking::NeedNewKeyFrame() {
    /// 如果用户在界面上选择重定位，那么将不插入关键帧
    /// 因为插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if (mbOnlyTracking)
        return false;

    /// 如果局部地图被闭环检测使用，则不插入关键帧
    if (mpLocalMapper->IsStopped() || mpLocalMapper->StopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    /// 如果重定位后没有超过一定帧数,或者关键帧数量不够,则不插入关键帧
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    /// 得到参考关键帧跟踪到的MapPoints数量
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    /// 获取地图点的数目,参考帧观测的数目大于等于 nMinObs
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    /// 查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    /// 对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
    int nMap = 0;           //现有地图中,可以被关键帧观测到的地图点数目
    int nTotal = 0;         //当前帧中可以添加到地图中的地图点数量
    if (mSensor != System::MONOCULAR) {
        for (int i = 0; i < mCurrentFrame.N; i++) {
            /// 如果是近点,并且这个特征点的深度合法,就可以被添加到地图中
            if (mCurrentFrame.mvDepth[i] > 0
                && mCurrentFrame.mvDepth[i] < mThDepth) {
                nTotal++;   // 总的可以添加mappoints数
                if (mCurrentFrame.mvpMapPoints[i])
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        nMap++;// 被关键帧观测到的mappoints数，即观测到地图中的MapPoints数量
            }
        }
    } else {
        nMap = 1;
        nTotal = 1;
    }

    /// 计算当前帧中观测到的地图点数目和当前帧中总共的地图点数目之比
    /// 这个值越接近1越好,越接近0说明跟踪上的地图点太少,tracking is weak
    const float ratioMap = (float)nMap / (float)(std::max(1, nTotal));

    /// 决策是否需要插入关键帧
    /// 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;  // 关键帧只有一帧，那么插入关键帧的阈值设置很低
    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;  //单目情况下插入关键帧的阈值很高

    /// MapPoints中和地图关联的比例阈值
    float thMapRatio = 0.35f;
    if (mnMatchesInliers > 300)
        thMapRatio = 0.20f;

    /// 超过MaxFrames个帧还没有有插入关键帧
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    /// 超过MinFrames帧没有插入关键帧,localMapper处于空闲状态
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames
                      && bLocalMappingIdle);
    /// 跟踪比较差
    const bool c1c =
            mSensor != System::MONOCULAR &&
                    (mnMatchesInliers < nRefMatches * 0.25 ||  ratioMap < 0.3f);
    /// 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio ||
                        ratioMap < thMapRatio) && mnMatchesInliers > 15);

    if ((c1a || c1b || c1c) && c2) {
        /// 如果地图接收关键帧,可以插入,柔则发送信号停止BA
        if (bLocalMappingIdle) {
            /// 可以插入关键帧
            return true;
        } else {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR) {
                /// 队列里不能阻塞太多关键帧
                /// tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                /// 然后localmapper再逐个pop出来插入到mspKeyFrames
                if (mpLocalMapper->KeyframesInQueue() < 3 )
                    /// 队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    /// 队列中缓冲的关键帧数目太多,暂时不能插入
                    return false;
            } else
                // 对于单目情况,就直接无法插入关键帧了
                return false;
        }
    } else
        /// 不满足上面的条件,自然不能插入关键帧
        return false;
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 */
void Tracking::CreateNewKeyFrame() {
    /// 如果不能保持局部建图器开启的状态,就无法顺利插入关键帧
    if (!mpLocalMapper->SetNotStop(true))
        return;

    /// 将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    /// 将当前关键帧设置为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    /// 对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if (mSensor != System::MONOCULAR) {
        /// 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        /// 根据相机深度排序,得出小于阈值的特征点,构造至少100个
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty()) {
            /// 按照深度从小到大排序
            sort(vDepthIdx.begin(), vDepthIdx.end());

            /// 将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for(size_t j = 0; j < vDepthIdx.size(); j++) {
                int i = vDepthIdx[j].second;
                bool bCreateNew = false;
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                /// 如果当前帧中无这个地图点,创建临时点
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1) { // 如果被观测数量小于1,创建临时点
                    //或者是刚刚创立
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
                }

                /// 如果需要新建地图点.这里是实打实的在全局地图中新建地图点
                if (bCreateNew) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    /// 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                } else {
                    nPoints++;
                }

                /// 当当前处理的点大于深度阈值或者已经处理的点超过阈值的时候,就不再进行了
                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    /// 执行插入关键帧的操作,其实也是在列表中等待
    mpLocalMapper->InsertKeyFrame(pKF);

    /// 然后现在允许局部建图器停止了
    mpLocalMapper->SetNotStop(false);

    /// 当前帧成为新的关键帧
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 对 Local MapPoints 进行跟踪
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 * Step 1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
 * Step 2：将所有 局部MapPoints 投影到当前帧，判断是否在视野范围内
 * Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配
 */
void Tracking::SearchLocalPoints() {
    /// 遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    /// 因为当前的mvpMapPoints一定在当前帧的视野中
    for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(),
                 vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++) {
        MapPoint* pMP = *vit;
        if (pMP) {
            if (pMP->isBad()) {
                *vit = static_cast<MapPoint*>(nullptr);
            } else {
                /// 更新能观测到该点的帧数加1(被当前帧看到了)
                pMP->IncreaseVisible();
                /// 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                /// 标记该点将来不被投影，因为已经匹配过(指的是使用恒速运动模型进行投影)
                pMP->mbTrackInView = false;
            }
        }
    }
    /// 准备进行投影匹配的点的数目
    int nToMatch = 0;

    /// 将所有局部MapPoints投影到当前帧，判断是否在视野范围内
    for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                 vend = mvpLocalMapPoints.end(); vit != vend; vit++) {
        MapPoint* pMP = *vit;
        /// 已经被当前帧观测到的MapPoint不再需要判断是否能被当前帧观测到
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        /// 跳过局部地图中的坏点
        if (pMP->isBad())
            continue;
        
        /// 判断LocalMapPoints中的点是否在在视野内
        if (mCurrentFrame.IsInFrustum(pMP, 0.5)) {
        	/// 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            pMP->IncreaseVisible();
            /// 只有在视野范围内的MapPoints才参与之后的投影匹配
            nToMatch++;
        }
    }

    /// 如果需要进行投影匹配的点的数目大于0，就进行投影匹配
    if (nToMatch > 0) {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)   /// RGBD相机时,搜索的阈值会变得稍微大一些
            th = 3;

        /// 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;

        /// 对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

/**
 * @brief 更新局部地图 LocalMap
 * 局部地图包括：共视关键帧、临近关键帧及其子父关键帧，由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap() {
    /// 设置参考地图点用于绘图显示局部地图点（红色）
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    /// 更新局部关键帧和局部MapPoints
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 更新局部地图点（来自局部关键帧）
 */
void Tracking::UpdateLocalPoints() {
    /// 清空局部MapPoints
    mvpLocalMapPoints.clear();

    /// 遍历局部关键帧mvpLocalKeyFrames
    for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                 itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++) {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        /// 将局部关键帧的MapPoints添加到mvpLocalMapPoints
        for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                     itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {
            MapPoint* pMP = *itMP;
            if (!pMP)
                continue;
            /// 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
            /// 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad()) {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 更新局部关键帧
 * 方法是遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
 * Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧 
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
 * Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
 * Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
 */
void Tracking::UpdateLocalKeyFrames() {
    /// 遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
    map<KeyFrame*,int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad()) {
                /// 观测到该MapPoint的KF和该MapPoint在KF中的索引
                const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                /// 这里一个地图点可以被多个关键帧观测到,因此对于每一次观测,都获得观测到这个地图点的关键帧,并且对关键帧进行投票
                for (map<KeyFrame*,size_t>::const_iterator it = observations.begin(),
                             itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            } else {
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }

    /// 意味着没有任何一个关键这观测到当前的地图点
    if (keyframeCounter.empty())
        return;

    /// 存储具有最多观测次数（max）的关键帧
    int max = 0;
    KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);

    /// 更新局部关键帧, 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    /// 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧
    for (map<KeyFrame*,int>::const_iterator it = keyframeCounter.begin(),
                 itEnd = keyframeCounter.end(); it != itEnd; it++) {
        KeyFrame* pKF = it->first;

        /// 如果设定为要删除的，跳过
        if (pKF->isBad())
            continue;
        
        /// 更新具有最大观测是-护目的关键帧
        if (it->second > max) {
            max = it->second;
            pKFmax = pKF;
        }
        /// 添加到局部关键帧的列表里
        mvpLocalKeyFrames.push_back(it->first);
        
        /// 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        /// 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    /// 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们作为局部关键帧
    for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                 itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++) {
        /// 局部关键帧不超过80帧
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame* pKF = *itKF;

        /// 最佳共视的10帧; 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧（将邻居的邻居拉拢入伙）
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                     itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++) {
            KeyFrame* pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad()) {
                /// mnTrackReferenceForFrame防止重复添加局部关键帧
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        /// 将自己的子关键帧作为局部关键帧（将邻居的子孙们拉拢入伙）
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for (set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                     send = spChilds.end(); sit != send; sit++) {
            KeyFrame* pChildKF = *sit;
            if (!pChildKF->isBad()) {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        /// 自己的父关键帧（将邻居的父母们拉拢入伙）
        KeyFrame* pParent = pKF->GetParent();
        if (pParent) {
            /// mnTrackReferenceForFrame防止重复添加局部关键帧
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }

    }

    /// 更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @details 重定位过程
 *     Step 1：计算当前帧特征点的Bow映射
 *     Step 2：找到与当前帧相似的候选关键帧
 *     Step 3：通过BoW进行匹配
 *     Step 4：通过EPnP算法估计姿态
 *     Step 5：通过PoseOptimization对姿态进行优化求解
 *     Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization() {
    /// 计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    /// 跟踪丢失的时候,找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    
    /// 如果没有候选关键帧，则退出
    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    ORBmatcher matcher(0.75, true);
    /// 每个关键帧的解算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    ///每个关键帧和当前帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    
    /// 放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    ///有效的候选关键帧数目
    int nCandidates = 0;

    /// 遍历所有的候选关键帧
    for (int i = 0; i < nKFs; i++) {
        KeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else {
            /// 通过BoW进行匹配
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if(nmatches < 15) { /// 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
                vbDiscarded[i] = true;
                continue;
            } else {
                /// 初始化PnPsolver
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                    0.99,   //用于计算RANSAC迭代次数理论值的概率
                    10,     //最小内点数, 注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                    300,    //最大迭代次数
                    4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                    0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991); //自由度为2的卡方检验的阈值,作为内外点判定时的距离的baseline(程序中还会根据特征点所在的图层对这个阈值进行缩放的)
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    /// 使用p4p ransac,直到找到足够多的内点
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    /// 通过一系列骚操作,直到找到能够进行重定位的匹配上的关键帧
    while (nCandidates > 0 && !bMatch) {
        /// 遍历当前所有的候选关键帧
        for (int i = 0; i < nKFs; i++) {
            /// 如果是需要剔除的点则继续
            if (vbDiscarded[i])
                continue;
            /// 内点标记
            vector<bool> vbInliers;     
            
            /// 内点数
            int nInliers;
            
            /// 表示RANSAC已经没有更多的迭代次数可用
            bool bNoMore;

            /// 通过EPnP算法估计姿态
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            /// 如果RANSAC达到最大迭代次数,则放弃关键帧
            if (bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            /// 如果相机位姿计算得出,进行优化
            if (!Tcw.empty()) {
                Tcw.copyTo(mCurrentFrame.mTcw);
                
                /// 成功被再次找到的地图点的集合,其实就是经过RANSAC之后的内点
                set<MapPoint*> sFound;

                const int np = vbInliers.size();
                /// 遍历所有内点
                for (int j = 0; j < np; j++) {
                    if (vbInliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    } else
                        mCurrentFrame.mvpMapPoints[j] = nullptr;
                }

                /// 通过PoseOptimization对姿态进行优化求解,只优化位姿,不优化地图点的坐标,返回的是内点的数量
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                /// 如果优化之后的内点数目不多,跳过本次循环
                if (nGood < 10)
                    continue;

                /// 删除外点对应的地图点
                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(nullptr);

                /// 如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                /// 前面的匹配关系是用词袋匹配过程得到的
                if (nGood < 50) {
                    int nadditional = matcher2.SearchByProjection(
                        mCurrentFrame,          //当前帧
                        vpCandidateKFs[i],      //关键帧
                        sFound,                 //已经找到的地图点集合
                        10,                     //窗口阈值
                        100);                   //ORB描述子距离

                    /// 如果通过投影过程获得了比较多的特征点
                    if (nadditional+nGood >= 50) {
                        /// 根据投影匹配的结果，采用3D-2D pnp非线性优化求解
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        /// 如果这样依赖内点数还是比较少的话,就使用更小的窗口搜索投影点
                        if (nGood > 30 && nGood < 50) {
                            /// 重新进行搜索
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(
                                mCurrentFrame,          //当前帧
                                vpCandidateKFs[i],      //候选的关键帧
                                sFound,                 //已经找到的地图点
                                3,                      //新的窗口阈值
                                64);                    //ORB描述子距离

                            /// Final optimization
                            if (nGood + nadditional >= 50) {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                /// 更新地图点
                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = nullptr;
                            }
                        }
                    }
                }

                /// 如果对于当前的关键帧已经有足够的内点(50个)了,那么就认为当前的这个关键帧已经和当前帧匹配上了
                if (nGood >= 50) {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    /// 如果没有匹配上
    if (!bMatch) {
        return false;
    } else {
        /// 如果匹配上了,说明当前帧重定位成功了.因此记录当前帧的id
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

/// 整个追踪线程执行复位操作
void Tracking::Reset() {
    /// 逐个请求其他线程停止
    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    /// 重启局部建图线程
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    /// 重启回环检测线程
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    /// 清除词袋数据库
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    /// 清除地图
    mpMap->clear();

    /// 然后复位各种状态变量
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer) {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(nullptr);
    }

    /// 清空各种链表
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
        mpViewer->Release();
}

/// 根据配置文件中的参数重新改变已经设置在系统中的参数
void Tracking::ChangeCalibration(const string& strSettingPath) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    /// 有些相机没有k3参数
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);
    mbf = fSettings["Camera.bf"];
    /// 设置重新进行计算标志为true
    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool& flag) {
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM