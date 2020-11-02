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
#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {
    
/// 网格的行数
#define FRAME_GRID_ROWS 48
/// 网格的列数
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
    
class Frame {
public:
	
    /// 默认构造函数
    Frame();

    /// 拷贝构造函数
    Frame(const Frame& frame);
    
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
    Frame(const cv::Mat& imLeft, const cv::Mat& imRight, 
          const double& timeStamp, ORBextractor* extractorLeft,
          ORBextractor* extractorRight, ORBVocabulary* voc, 
          cv::Mat& K, cv::Mat& distCoef, const float& bf, const float& thDepth);

    /**
     * @brief 为RGBD相机准备的帧构造函数
     * @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
     * @param[in] imDepth       深度图像
     * @param[in] timeStamp     时间戳
     * @param[in] extractor     特征点提取器句柄
     * @param[in] voc           ORB特征点词典的句柄
     * @param[in] K             相机的内参数矩阵
     * @param[in] distCoef      相机的去畸变参数
     * @param[in] bf            baseline*bf
     * @param[in] thDepth       远点和近点的深度区分阈值
     */
    Frame(const cv::Mat& imGray, const cv::Mat& imDepth, const double& timeStamp,
          ORBextractor* extractor,ORBVocabulary* voc, cv::Mat& K,
          cv::Mat& distCoef, const float& bf, const float& thDepth);

    /**
     * @brief 为单目相机准备的帧构造函数
     * @param[in] imGray                            //灰度图
     * @param[in] timeStamp                         //时间戳
     * @param[in & out] extractor                   //ORB特征点提取器的句柄
     * @param[in] voc                               //ORB字典的句柄
     * @param[in] K                                 //相机的内参数矩阵
     * @param[in] distCoef                          //相机的去畸变参数
     * @param[in] bf                                //baseline*f
     * @param[in] thDepth                           //区分远近点的深度阈值
     */
    Frame(const cv::Mat& imGray, const double& timeStamp, ORBextractor* extractor,
          ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef,
          const float& bf, const float& thDepth);

    /**
     * @brief 提取图像的ORB特征，提取的关键点存放在mvKeys，描述子存放在mDescriptors
     * @param[in] flag          标记是左图还是右图。0：左图  1：右图
     * @param[in] im            等待提取特征点的图像
     */
    void ExtractORB(int flag, const cv::Mat& im);

    /**
     * @brief 计算词袋模型 
     * 如果没有传入已有的词袋数据，则就用当前的描述子重新计算生成词袋数据
     */
    void ComputeBoW();

    /// 设置相机位姿,并计算光心位置
    void SetPose(cv::Mat Tcw);

    /// 根据Tcw计算mRcw、mtcw和mRwc、mOw
    void UpdatePoseMatrices();

    /// 返回相机光心位置
    inline cv::Mat GetCameraCenter() const {
        return mOw.clone();
    }

    /// 返回旋转的逆矩阵
    inline cv::Mat GetRotationInverse() const {
        return mRwc.clone();
    }

    /// 判断地图点是否在当前帧的视野中,同时设置地图点变量给跟踪来用
    bool IsInFrustum(MapPoint* pMP, float viewingCosLimit);

    /**
     * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
     * 
     * @param[in] kp                    给定的特征点
     * @param[in & out] posX            特征点所在网格坐标的横坐标
     * @param[in & out] posY            特征点所在网格坐标的纵坐标
     * @return true                     如果找到特征点所在的网格坐标，返回true
     * @return false                    没找到返回false
     */
    bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

    /// 获取特定区域内的坐标点(x,y为中心,半径为r区域,[minLevel, maxLevel]的特征点
    vector<size_t> GetFeaturesInArea(const float& x, const float& y,
                                     const float& r, const int minLevel = -1,
                                     const int maxLevel = -1) const;

    /// 从双目信息中恢复深度 (搜索从右图和左图的匹配,用于计算深度,并存储右图关键点)
    /// 为左图的每一个特征点在右图中找到匹配点,根据基线(有冗余范围)上描述子距离找到匹配
    /// 进行SAD精确定位，最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对
    /// 然后利用抛物线拟合得到亚像素精度的匹配，匹配成功后会更新 mvuRight 和 mvDepth
    void ComputeStereoMatches();

    /// 从RGBD相机获得深度
    void ComputeStereoFromRGBD(const cv::Mat& imDepth);

    /// 计算特征点在三维空间的坐标
    cv::Mat UnprojectStereo(const int& i);

public:

    // Vocabulary used for relocalization.
    ///用于重定位的ORB特征字典
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ///ORB特征提取器句柄,其中右侧的提取器句柄只会在双目输入的情况中才会被用到
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    ///帧的时间戳
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.

    /**
     * @name 相机的内参数
     * @{
     */

    ///相机的内参数矩阵
    cv::Mat mK;
	//NOTICE 注意这里的相机内参数其实都是类的静态成员变量；此外相机的内参数矩阵和矫正参数矩阵却是普通的成员变量，
	//NOTE 这样是否有些浪费内存空间？

    
    static float fx;        ///<x轴方向焦距
    static float fy;        ///<y轴方向焦距
    static float cx;        ///<x轴方向光心偏移
    static float cy;        ///<y轴方向光心偏移
    static float invfx;     ///<x轴方向焦距的逆
    static float invfy;     ///<x轴方向焦距的逆

	//TODO 目测是opencv提供的图像去畸变参数矩阵的，但是其具体组成未知
    ///去畸变参数
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    ///baseline x fx
    float mbf;

    // Stereo baseline in meters.
    ///相机的基线长度,单位为米
    float mb;

    /** @} */

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
	//TODO 这里它所说的话还不是很理解。尤其是后面的一句。
    //而且,这个阈值不应该是在哪个帧中都一样吗?
    ///判断远点和近点的深度阈值
    float mThDepth;

    /// 特征点数量
    int N; 

    /**
     * @name 关于特征点
     * @{ 
     */

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    
    ///原始左图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeys;
    ///原始右图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeysRight;
	///校正mvKeys后的特征点
    std::vector<cv::KeyPoint> mvKeysUn;

    

    ///@note 之所以对于双目摄像头只保存左图像矫正后的特征点,是因为对于双目摄像头,一般得到的图像都是矫正好的,这里再矫正一次有些多余.\n
    ///校正操作是在帧的构造函数中进行的。
    
    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1

    /// 对于单目摄像头，这两个容器中存的都是-1
    /// 对于双目相机,存储左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）
    std::vector<float> mvuRight;	//m-member v-vector u-指代横坐标,因为最后这个坐标是通过各种拟合方法逼近出来的，所以使用float存储
    /// 对应的深度
    std::vector<float> mvDepth;
    
    // Bag of Words Vector structures.
    ///和词袋模型有关的向量
    DBoW2::BowVector mBowVec;
    ///和词袋模型中特征有关的向量
    DBoW2::FeatureVector mFeatVec;
    ///@todo 这两个向量目前的具体含义还不是很清楚

    // ORB descriptor, each row associated to a keypoint.
    /// 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    /// 每个特征点对应的MapPoint.如果特征点没有对应的地图点,那么将存储一个空指针
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    /// 属于外点的特征点标记,在 Optimizer::PoseOptimization 使用了
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
	//原来通过对图像分区域还能够降低重投影地图点时候的匹配复杂度啊。。。。。
    ///@note 注意到上面也是类的静态成员变量， 有一个专用的标志mbInitialComputations用来在帧的构造函数中标记这些静态成员变量是否需要被赋值
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementHeightInv;

    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
	/// 这个向量中存储的是每个图像网格内特征点的id（左图）
    /// 相当于三维数组,第三维表示
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    /// 相机位姿
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

    /// 类的静态成员变量，这些变量则是在整个系统开始执行的时候被初始化的——它在全局区被初始化
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    /// 参考关键帧指针
    KeyFrame* mpReferenceKF;

    /**
     * @name 图像金字塔信息
     * @{
     */
    // Scale pyramid info.
    int mnScaleLevels;                  ///<图像金字塔的层数
    float mfScaleFactor;                ///<图像金字塔的尺度因子
    float mfLogScaleFactor;             ///<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度
                                  
    vector<float> mvScaleFactors;		///<图像金字塔每一层的缩放因子
    vector<float> mvInvScaleFactors;	///<以及上面的这个变量的倒数
    vector<float> mvLevelSigma2;		///@todo 目前在frame.c中没有用到，无法下定论
    vector<float> mvInvLevelSigma2;		///<上面变量的倒数

    /** @} */

    // Undistorted Image Bounds (computed once).
    /**
     * @name 用于确定画格子时的边界 
     * @note（未校正图像的边界，只需要计算一次，因为是类的静态成员变量）
     * @{
     */
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    /** @} */

    /**
     * @brief 一个标志，标记是否已经进行了这些初始化计算
     * @note 由于第一帧以及SLAM系统进行重新校正后的第一帧会有一些特殊的初始化处理操作，所以这里设置了这个变量. \n
     * 如果这个标志被置位，说明再下一帧的帧构造函数中要进行这个“特殊的初始化操作”，如果没有被置位则不用。
    */ 
    static bool mbInitialComputations;

private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
	/**
     * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
     * 
     */
    void UndistortKeyPoints();

    /**
     * @brief 计算去畸变图像的边界
     * 
     * @param[in] imLeft            需要计算边界的图像
     */
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    /**
     * @brief 将提取到的特征点分配到图像网格中 \n
     * @details 该函数由构造函数调用
     * 
     */
    void AssignFeaturesToGrid();

    /**
     * @name 和相机位姿有关的变量
     * @{
     */
    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world

    /** @} */
};

}// namespace ORB_SLAM

#endif // FRAME_H
