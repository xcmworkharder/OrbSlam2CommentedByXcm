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
#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2 {
/// 为图像帧进行编号统一
long unsigned int Frame::nNextId = 0;

/// 初始化操作标志,最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧
bool Frame::mbInitialComputations = true;

/// 静态变量初始化,默认值为0
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

/// 无参的构造函数默认为空
Frame::Frame() {

}

/// 拷贝构造函数
Frame::Frame(const Frame& frame)
        : mpORBvocabulary(frame.mpORBvocabulary),
          mpORBextractorLeft(frame.mpORBextractorLeft),
          mpORBextractorRight(frame.mpORBextractorRight),
          mTimeStamp(frame.mTimeStamp),
          mK(frame.mK.clone()),
          mDistCoef(frame.mDistCoef.clone()),
          mbf(frame.mbf),
          mb(frame.mb),
          mThDepth(frame.mThDepth),
          N(frame.N),
          mvKeys(frame.mvKeys),
          mvKeysRight(frame.mvKeysRight),
          mvKeysUn(frame.mvKeysUn),
          mvuRight(frame.mvuRight),
          mvDepth(frame.mvDepth),
          mBowVec(frame.mBowVec),
          mFeatVec(frame.mFeatVec),
          mDescriptors(frame.mDescriptors.clone()),
          mDescriptorsRight(frame.mDescriptorsRight.clone()),
          mvpMapPoints(frame.mvpMapPoints),
          mvbOutlier(frame.mvbOutlier),
          mnId(frame.mnId),
          mpReferenceKF(frame.mpReferenceKF),
          mnScaleLevels(frame.mnScaleLevels),
          mfScaleFactor(frame.mfScaleFactor),
          mfLogScaleFactor(frame.mfLogScaleFactor),
          mvScaleFactors(frame.mvScaleFactors),
          mvInvScaleFactors(frame.mvInvScaleFactors),
          mvLevelSigma2(frame.mvLevelSigma2),
          mvInvLevelSigma2(frame.mvInvLevelSigma2) {

    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty())
        /// 这里是给新的帧设置Pose
        SetPose(frame.mTcw);
}


/**
* @brief 为双目相机准备的构造函数
* @param[in] imLeft            左目图像
* @param[in] imRight           右目图像
* @param[in] timeStamp         时间戳
* @param[in] extractorLeft     左目图像特征点提取器指针
* @param[in] extractorRight    右目图像特征点提取器指针
* @param[in] voc               ORB字典指针
* @param[in] K                 相机内参矩阵
* @param[in] distCoef          相机去畸变参数
* @param[in] bf                相机基线长度和焦距的乘积
* @param[in] thDepth           远点和近点的深度区分阈值
*
*/
Frame::Frame(const cv::Mat& imLeft, const cv::Mat& imRight,
             const double& timeStamp, ORBextractor* extractorLeft,
             ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef, const float& bf, const float& thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft),
          mpORBextractorRight(extractorRight), mTimeStamp(timeStamp),
          mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf),
          mThDepth(thDepth),
          mpReferenceKF(static_cast<KeyFrame*>(nullptr)) {
    /// 为新帧赋id号
    mnId = nNextId++;

    /// 计算图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    /// 这个是获得层与层之前的缩放比
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    /// 计算上面缩放比的对数,log=自然对数，log10=才是以10为基底的对数
    mfLogScaleFactor = log(mfScaleFactor);
    /// 获取每层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    /// 获取每层图像缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    /// 高斯模糊的时候，使用的方差
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    /// 获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    /// 同时对左右目提取ORB特征点
    thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
    thread threadRight(&Frame::ExtractORB, this, 1, imRight);
    /// 等待两张图像特征点提取过程完成
    threadLeft.join();
    threadRight.join();

    /// mvKeys存放提取的特征点,如果没有则退出
    N = mvKeys.size();

    /// 如果左图像中没有成功提取到特征点那么就返回，也意味这这一帧的图像无法使用
    if (mvKeys.empty())
        return;

    /// 对特征点进行畸变校正
    UndistortKeyPoints();

    /// 计算双目间特征点的匹配，匹配成功会计算其深度,深度存放在mvuRight和mvDepth中
    ComputeStereoMatches();

    /// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
    /// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N, false);

    /// 在第一次进入或者标定文件发生变化重新初始化的时候,重新计算相关相机参数
    if (mbInitialComputations) {
        /// 计算去畸变后图像的边界
        ComputeImageBounds(imLeft);

        /// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        /// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        /// 相机内参赋值
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        /// 初始化过程完成，标志复位
        mbInitialComputations = false;
    }

    /// 双目相机基线长度
    mb = mbf / fx;

    /// 将特征点分配到图像网格中,好处是可以设置网格内特征点上限,从而使得特征点分布更均匀
    AssignFeaturesToGrid();
}

/**
* @brief 为RGBD相机准备的帧构造函数
* @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
* @param[in] imDepth       深度图像
* @param[in] timeStamp     时间戳
* @param[in] extractor     特征点提取器指针
* @param[in] voc           ORB特征点词典的指针
* @param[in] K             相机的内参数矩阵
* @param[in] distCoef      相机的去畸变参数
* @param[in] bf            baseline*bf
* @param[in] thDepth       远点和近点的深度区分阈值
*/
Frame::Frame(const cv::Mat& imGray, const cv::Mat& imDepth,
             const double& timeStamp, ORBextractor* extractor,
             ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef,
             const float& bf, const float& thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
          mpORBextractorRight(static_cast<ORBextractor *>(nullptr)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()),
          mbf(bf), mThDepth(thDepth) {
    /// 帧序号在全局范围自增
    mnId = nNextId++;
    /// 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    /// 获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    /// 计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
    /// 获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    /// 获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    /// 计算上面获取的sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    /// 对图像进行提取特征点, 第一个参数0-左图， 1-右图
    ExtractORB(0, imGray);

    /// 获取特征点的个数
    N = mvKeys.size();

    /// 如果这一帧没有能够提取出特征点，那么就直接返回
    if (mvKeys.empty())
        return;

    /// 使用内参对提取到的特征点进行矫正
    UndistortKeyPoints();

    /// 根据像素坐标获取深度信息，如果存在则保存下来，这里还计算了假想右图的对应特征点的横坐标
    ComputeStereoFromRGBD(imDepth);

    /// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
    /// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N, false);

    /// 在第一次进入或者标定文件发生变化重新初始化的时候,重新计算相关相机参数
    if (mbInitialComputations) {
        /// 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        /// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        /// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        /// 给类的静态成员变量赋值
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        /// 初始化过程完成，标志复位
        mbInitialComputations = false;
    }

    /// 双目相机基线长度
    mb = mbf / fx;

    /// 将特征点分配到图像网格中,好处是可以设置网格内特征点上限,从而使得特征点分布更均匀
    AssignFeaturesToGrid();
}

/**
* @brief 单目帧构造函数
* @param[in] imGray                            //灰度图
* @param[in] timeStamp                         //时间戳
* @param[in & out] extractor                   //ORB特征点提取器的指针
* @param[in] voc                               //ORB字典的指针
* @param[in] K                                 //相机的内参数矩阵
* @param[in] distCoef                          //相机的去畸变参数
* @param[in] bf                                //baseline*f
* @param[in] thDepth                           //区分远近点的深度阈值
*/
Frame::Frame(const cv::Mat& imGray, const double& timeStamp,
             ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef, const float& bf, const float& thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
          mpORBextractorRight(static_cast<ORBextractor*>(nullptr)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()),
          mbf(bf), mThDepth(thDepth) {
    /// 帧序号在全局范围自增
    mnId = nNextId++;
    /// 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    /// 获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    /// 计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
    /// 获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    /// 获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    /// 获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    /// 获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    /// 对图像进行提取特征点, 第一个参数0-左图， 1-右图
    ExtractORB(0, imGray);
    /// 获取特征点的个数
    N = mvKeys.size();
    /// 如果这一帧没有能够提取出特征点，那么就直接返回
    if (mvKeys.empty())
        return;

    /// 使用内参对提取到的特征点进行矫正
    UndistortKeyPoints();

    /// 没有右目和深度,都赋值为-1
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);

    /// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
    /// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N, false);

    /// 在第一次进入或者标定文件发生变化重新初始化的时候,重新计算相关相机参数
    if (mbInitialComputations) {
        /// 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        /// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        /// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        /// 给类的静态成员变量赋值
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        /// 特殊的初始化过程完成，标志复位
        mbInitialComputations = false;
    }

    /// 计算 basline
    mb = mbf / fx;

    /// 将特征点分配到图像网格中,好处是可以设置网格内特征点上限,从而使得特征点分布更均匀
    AssignFeaturesToGrid();
}

/// 将提取的ORB特征点分配到图像网格中
void Frame::AssignFeaturesToGrid() {
    /// 给存储特征点的网格数组 Frame::mGrid 预分配空间0.5N
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    /// 开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    /// 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中
    for (int i = 0; i < N; i++) {
        /// 获取已经去畸变后的特征点
        const cv::KeyPoint& kp = mvKeysUn[i];
        /// 存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        /// 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            /// 如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/**
* @brief 封装opencv的ORB提取函数,增加一个flag,决定提取的是左目还是右目，从而调用不同的特征提取器
* @param[in] flag          标记是左图还是右图。0：左图  1：右图
* @param[in] im            等待提取特征点的图像
*/
void Frame::ExtractORB(int flag, const cv::Mat& im) {
    /// 根据左图还是右图选择,使用仿函数来完成
    if (flag == 0)
        (*mpORBextractorLeft)(im,                // 待提取特征点的图像
                              cv::Mat(),         // 掩摸图像, 实际没有用到
                              mvKeys,            // 输出变量，用于保存提取后的特征点
                              mDescriptors);     // 输出变量，用于保存特征点的描述子
    else
        (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
}

/// 设置相机姿态，随后会调用 UpdatePoseMatrices()来改变mRcw,mRwc等变量的值
void Frame::SetPose(cv::Mat Tcw) {
    /// 更改类的成员变量,深拷贝
    mTcw = Tcw.clone();
    /// 调用 Frame::UpdatePoseMatrices() 来更新、计算类的成员变量中所有的位姿矩阵
    UpdatePoseMatrices();
}

/// 根据Tcw计算mRcw、mtcw和mRwc、mOw
void Frame::UpdatePoseMatrices() {
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;
}

/**
* @brief 判断路标点是否在视野中
        先计算地图点在相机坐标系的坐标,用该点和光心连线知道相机的视角(连线和相机正前方夹角)
        如果大于设定角度,则该点不在视野范围,如果在计算在该帧图像上的坐标,以便跟踪使用
* @param[in] pMP                       当前地图点
* @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
* @return true                         地图点合格，且在视野内 地图点不合格，抛弃
*/
bool Frame::IsInFrustum(MapPoint* pMP, float viewingCosLimit) {
    /// 判断一个地图点是否进行投影的标志
    pMP->mbTrackInView = false;
    /// 获得这个地图点的世界坐标
    cv::Mat P = pMP->GetWorldPos();
    /// 3d点P在相机坐标系下的坐标
    const cv::Mat Pc = mRcw * P + mtcw;
    const float& PcX = Pc.at<float>(0);
    const float& PcY = Pc.at<float>(1);
    const float& PcZ = Pc.at<float>(2);

    /// 检查地图点的深度是否为正,如果负的,返回false
    if (PcZ < 0.0f)
        return false;

    /// 将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
    const float invz = 1.0f / PcZ;
    const float u = fx * PcX * invz + cx;
    const float v = fy * PcY * invz + cy;

    /// 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    /// 计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();

    /// 得到世界坐标系下地图点到光心的矢量
    const cv::Mat PO = P - mOw;
    /// 取模得到距离
    const float dist = cv::norm(PO);

    /// 如果不在允许的尺度变化范围内，认为重投影不可靠
    if (dist < minDistance || dist > maxDistance)
        return false;

    /// 获取当前视角和平均视角的夹角余弦值,
    cv::Mat Pn = pMP->GetNormal();  // 平均视角
    const float viewCos = PO.dot(Pn) / dist;

    /// 如果大于给定的阈值 cos(60°)=0.5，认为这个点方向太偏了，重投影不可靠，返回false
    if (viewCos < viewingCosLimit)
        return false;

    /// 根据地图点到光心的距离(深度)来预测一个尺度层级（仿照特征点金字塔层级）
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    /// 为跟踪设置一些参数
    pMP->mbTrackInView = true;
    /// 该地图点投影在当前图像（一般是左图）的像素横坐标
    pMP->mTrackProjX = u;
    /// bf/z其实是视差，相减得到右图（如有）中对应点的横坐标
    pMP->mTrackProjXR = u - mbf * invz;
    /// 该地图点投影在当前图像（一般是左图）的像素纵坐标
    pMP->mTrackProjY = v;
    /// 根据地图点到光心距离，预测的该地图点的尺度层级
    pMP->mnTrackScaleLevel = nPredictedLevel;
    /// 保存当前视角和法线夹角的余弦值
    pMP->mTrackViewCos = viewCos;

    /// 执行到这里说明这个地图点在相机的视野中并且进行投影是可靠的，返回true
    return true;
}

/**
* @brief 找到以x,y为中心,半径为r区域,且金字塔层级在[minLevel, maxLevel]的特征点
* @param[in] x                     特征点坐标x
* @param[in] y                     特征点坐标y
* @param[in] r                     搜索半径
* @param[in] minLevel              最小金字塔层级
* @param[in] maxLevel              最大金字塔层级
* @return vector<size_t>           返回搜索到的候选匹配点id
*/
vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y,
                                        const float& r, const int minLevel,
                                        const int maxLevel) const {
    vector<size_t> vIndices;
    vIndices.reserve(N);

    /// 左边界网格列索引
    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    /// 如果左边界超过了设定了上限，说明方形内没有特征点
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;
    /// 右边界网格列索引
    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int) ceil(
            (x - mnMinX + r) * mfGridElementWidthInv));
    /// 右边界网格列索引小于下限, 说明没有特征点
    if (nMaxCellX < 0)
        return vIndices;
    /// 以下是行信息
    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;
    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1,
                              (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    /// 图像金字塔范围是否正常
    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    /// 区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            /// 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            /// 遍历网格中特征点,存储有效的特征点
            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
                /// 搜索金字塔层级范围要合法
                if (bCheckLevels) {
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }

                /// 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                /// 记录在半径为r的圆内特征点
                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}


/**
* @brief 计算某个特征点所在网格的网格坐标
* @param[in] kp                    给定的特征点
* @param[in & out] posX            特征点所在网格坐标的横坐标
* @param[in & out] posY            特征点所在网格坐标的纵坐标
* @return                          如果找到特征点所在的网格坐标，返回true
*/
bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
    /// 计算特征点x,y坐标落在哪个网格内，网格坐标为posX，posY
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);
    /// 如果网格坐标posX，posY超出了[0,FRAME_GRID_COLS]和[0,FRAME_GRID_ROWS]范围
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;
    /// 计算成功返回true
    return true;
}

/**
 * @brief 计算词袋模型
 *      如果没有传入已有的词袋数据，则就用当前的描述子重新计算生成词袋数据
 */
void Frame::ComputeBoW() {
    if (mBowVec.empty()) {
        /// 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        /// 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        mpORBvocabulary->transform(vCurrentDesc,    // 当前的描述子vector
                                   mBowVec,         // 输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                                   mFeatVec,        // 输出，记录node id及其对应的图像 feature对应的索引
                                   4);              // 4表示从叶节点向前数的层数
    }
}

/// 用内参对特征点去畸变,结果保存到mvKeysUn中
void Frame::UndistortKeyPoints() {
    /// 变量mDistCoef中存储了opencv指定格式的去畸变参数 格式为：(k1,k2,p1,p2,k3)
    /// 如果第一个畸变参数k1为0，其余都是0,不需要校正
    if (mDistCoef.at<float>(0) == 0.0) {
        mvKeysUn = mvKeys;
        return;
    }

    /// N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N, 2, CV_32F);
    /// 遍历每个特征点，并将它们的坐标保存到矩阵中
    for (int i = 0; i < N; i++) {
        /// 然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    /// 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    mat = mat.reshape(2);       // reshape(int cn,int rows=0) cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    cv::undistortPoints(
            mat,                // 输入的特征点坐标
            mat,                // 输出的校正后的特征点坐标覆盖原矩阵
            mK,                 // 相机的内参数矩阵
            mDistCoef,          // 相机畸变参数矩阵
            cv::Mat(),          // 一个空矩阵，对应为函数原型中的R
            mK);                // 新内参数矩阵，对应为函数原型中的P

    /// 调整回只有一个通道，回归我们正常的处理方式
    mat = mat.reshape(1);

    /// 存储校正后的特征点
    mvKeysUn.resize(N);
    /// 遍历每一个特征点
    for (int i = 0; i < N; i++) {
        /// 根据索引获取这个特征点
        /// 这样做而不是直接重新声明一个特征点对象的目的是能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
        /// 读取校正后的坐标并覆盖老坐标
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

/**
* 计算去畸变图像的边界
*   @param[in] imLeft  需要计算边界的图像
*/
void Frame::ComputeImageBounds(const cv::Mat& imLeft) {
    /// 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if (mDistCoef.at<float>(0) != 0.0) {
        /// 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;         //左上
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols; //右上
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;         //左下
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols; //右下
        mat.at<float>(3, 1) = imLeft.rows;

        /// 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        /// 校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));// 左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));// 右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));// 左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));// 左下和右下纵坐标最小的
    } else {
        /// 如果畸变参数为0，就直接获得图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * 计算双目匹配信息,并恢复深度
 *    为左图的每一个特征点在右图中找到匹配点,根据基线(有冗余范围)上描述子距离找到匹配
 *    进行SAD精确定位，最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对
 *    然后利用抛物线拟合得到亚像素精度的匹配，匹配成功后会更新 mvuRight 和 mvDepth
 */
void Frame::ComputeStereoMatches() {
    /// 先初始化为-1
    mvuRight = vector<float>(N, -1.0f);
    mvDepth = vector<float>(N, -1.0f);

    /// orb特征相似度阈值
    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    /// 金字塔顶层（0层）图像高 nRows
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    /// 存储金字塔顶层的特征信息
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());
    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(200); // 进行预留,提高效率

    /// 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();

    /// 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for (int iR = 0; iR < Nr; iR++) {
        /// 获取特征点ir的y坐标，即行号
        const cv::KeyPoint& kp = mvKeysRight[iR];
        const float& kpY = kp.pt.y;

        /// 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大
        /// 如果特征点在金字塔第一层，则搜索范围为:正负2
        /// 尺度越大其位置不确定性越高，所以其搜索半径越大
        const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        /// 将特征点ir保证在可能的行号中
        for (int yi = minr; yi <= maxr; yi++)
            vRowIndices[yi].push_back(iR);
    }

    /// 设置搜索限制
    const float minZ = mb;          // 确实是bug, mb在该函数第一次调用时还没有初始化
    const float minD = 0;           // 最小视差
    const float maxD = mbf / minZ;  // 最大视差

    /// 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(N);

    /// 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for (int iL = 0; iL < N; iL++) {
        const cv::KeyPoint& kpL = mvKeys[iL];
        const int& levelL = kpL.octave;
        const float& vL = kpL.pt.y;
        const float& uL = kpL.pt.x;

        /// 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t>& vCandidates = vRowIndices[vL];
        if (vCandidates.empty()) continue;

        /// 计算理论上的最佳搜索范围
        const float minU = uL - maxD;
        const float maxU = uL - minD;

        /// 最大搜索范围小于0，说明无匹配点
        if (maxU < 0) continue;

        /// 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;
        const cv::Mat &dL = mDescriptors.row(iL);

        /// 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
        for (size_t iC = 0; iC < vCandidates.size(); iC++) {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint& kpR = mvKeysRight[iR];

            /// 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;
            /// 使用列坐标(x)进行匹配，和stereomatch一样
            const float& uR = kpR.pt.x;

            /// 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if (uR >= minU && uR <= maxU) {
                /// 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                /// 统计最小相似度及其对应的列坐标(x)
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        /// 精确匹配.
        if (bestDist < thOrbDist) {
            /// 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];

            /// 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x * scaleFactor);
            const float scaledvL = round(kpL.pt.y * scaleFactor);
            const float scaleduR0 = round(uR0 * scaleFactor);

            /// 滑动窗口搜索, 类似模版卷积或滤波,w表示sad相似度的窗口半径
            const int w = 5;

            /// 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(
                    scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
            IL.convertTo(IL, CV_32F);

            /// 图像块均值归一化，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

            /// 初始化最佳相似度
            int bestDist = INT_MAX;

            /// 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;

            /// 滑动窗口的滑动范围为（-L, L）
            const int L = 5;

            /// 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            /// 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            /// 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
            /// 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            /// 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;

            /// 判断搜索是否越界
            if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            /// 在搜索范围内从左到右滑动，并计算图像块相似度
            for (int incR = -L; incR <= +L; incR++) {
                /// 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(
                        scaledvL - w, scaledvL + w + 1).colRange(
                        scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                IR.convertTo(IR, CV_32F);
                /// 图像块均值归一化，降低亮度变化对相似度计算的影响
                IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);
                /// sad 计算
                float dist = cv::norm(IL, IR, cv::NORM_L1);
                /// 统计最小sad和偏移量
                if (dist < bestDist) {
                    bestDist = dist;
                    bestincR = incR;
                }
                /// L+incR 为refine后的匹配点列坐标(x)
                vDists[L + incR] = dist;
            }

            /// 搜索窗口越界判断
            if (bestincR == -L || bestincR == L)
                continue;

            /// 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            /// 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];
            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            /// 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if (deltaR < -1 || deltaR > 1)
                continue;

            /// 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave] * (scaleduR0 + (float) bestincR + deltaR);
            float disparity = (uL - bestuR);
            if (disparity >= minD && disparity < maxD) {
                /// 如果存在负视差，则约束为0.01
                if (disparity <= 0) {
                    disparity = 0.01;
                    bestuR = uL - 0.01;
                }

                // 根据视差值计算深度信息
                // 保存最相似点的列坐标(x)信息
                // 保存归一化sad最小相似度
                // 最优视差值/深度选择.
                mvDepth[iL] = mbf / disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int, int>(bestDist, iL));
            }
        }

        /// 删除outliers
        /// 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
        /// 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--) {
            if (vDistIdx[i].first < thDist)
                break;
            else {
                // 误匹配点置为-1，和初始化时保持一直，作为error code
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }
}

/// 从RGBD相机获得深度
void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth) {   //参数是深度图像
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);

    /// 开始遍历彩色图像中的所有特征点
    for (int i = 0; i < N; i++) {
        const cv::KeyPoint& kp = mvKeys[i];
        const cv::KeyPoint& kpU = mvKeysUn[i];

        /// 获取其横纵坐标
        const float& v = kp.pt.y;
        const float& u = kp.pt.x;
        /// 从深度图像中获取这个特征点对应的深度点
        const float d = imDepth.at<float>(v, u);

        if (d > 0) {
            /// 那么就保存这个点的深度
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x - mbf / d;
        }
    }
}

/// 将特征点坐标反投影到3D地图点(世界坐标)
cv::Mat Frame::UnprojectStereo(const int& i) {
    /// 获取深度信息
    const float z = mvDepth[i];
    /// 如果深度大于0
    if (z > 0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        /// 计算在当前相机坐标系下的坐标
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        /// 生成三维点（在当前相机坐标系下）
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}
} // namespace ORB_SLAM
