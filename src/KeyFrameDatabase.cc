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

#include "KeyFrameDatabase.h"
#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

using namespace std;


namespace ORB_SLAM2 {

/// 构造函数
KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary& voc) : mpVoc(&voc) {
    /// 使用单词数量为倒排索引设置大小
    mvInvertedFile.resize(voc.size());
}

/// 根据关键帧的词包，更新数据库的倒排索引
void KeyFrameDatabase::add(KeyFrame* pKF) {
    unique_lock<mutex> lock(mMutex);
    /// 为每一个word添加该KeyFrame
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
         vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

/// 关键帧被删除后，更新数据库的倒排索引
void KeyFrameDatabase::erase(KeyFrame* pKF) {
    unique_lock<mutex> lock(mMutex);
    /// 每一个KeyFrame包含多个words，遍历mvInvertedFile中words，然后在word中删除该KeyFrame
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                 vend = pKF->mBowVec.end(); vit != vend; vit++) {
        list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];
        for (list<KeyFrame*>::iterator lit = lKFs.begin(),
                     lend = lKFs.end(); lit != lend; lit++) {
            if (pKF == *lit) {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

/// 清空关键帧数据库
void KeyFrameDatabase::clear() {
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());// mpVoc：预先训练好的词典
}

/**
 * @brief 闭环检测找到与该关键帧可能存在闭环的关键帧
 *          1.找出和当前帧具有公共单词的所有关键帧（不包括与当前帧相连(附近)的关键帧）
 *          2.统计所有闭环候选帧中与pKF具有共同单词最多的单词数，
 *            只考虑共有单词数大于0.8*maxCommonWords以及匹配得分大于给定的minScore的关键帧，存入lScoreAndMatch
 *          3.对于筛选出来的pKFi，每一个都要抽取出自身的共视（共享地图点最多的前10帧）关键帧分为一组，
 *            计算该组整体得分（与pKF比较的），记为bestAccScore。所有组得分大于0.75*bestAccScore的，均当作闭环候选帧
 * @param pKF      需要闭环的关键帧
 * @param minScore 相似性分数最低要求
 * @return         可能闭环的所有关键帧
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore) {
    /// 获取所有与该pKF相连的KeyFrame，这些相连Keyframe都是局部相连，在闭环检测的时候将被剔除
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    /// 用于保存可能与pKF形成回环的候选帧（满足有相同的word，且不属于局部相连帧）
    list<KeyFrame*> lKFsSharingWords;
    /// 1.找出和当前帧具有公共单词的所有关键帧（不包括与当前帧链接的关键帧）
    {
        unique_lock<mutex> lock(mMutex);
        /// 遍历该pKF的每一个word
        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                     vend = pKF->mBowVec.end(); vit != vend; vit++) {
            /// 使用倒排索引容器提取所有包含该word的KeyFrame
            list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];
            /// 然后对这些关键帧展开遍历
            for (list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
                 lit != lend; lit++) {
                KeyFrame* pKFi = *lit;
                if (pKFi->mnLoopQuery != pKF->mnId) {       // pKFi还没有标记为pKF的候选帧
                    pKFi->mnLoopWords = 0;
                    if (!spConnectedKeyFrames.count(pKFi)) {// 与pKF局部链接的关键帧不进入闭环候选帧
                        pKFi->mnLoopQuery = pKF->mnId;      // pKFi标记为pKF的候选帧，之后直接跳过判断
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;    // 记录pKFi与pKF具有相同word的个数
            }
        }
    }

    /// 如果没有关键帧和这个关键帧具有相同的单词,那么就返回空
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*>> lScoreAndMatch;

    /// 2.统计所有闭环候选帧与pKF具有共同单词数量最多的数量,最大值存入maxCommonWords
    int maxCommonWords = 0;
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                 lend = lKFsSharingWords.end(); lit != lend; lit++) {
        if ((*lit)->mnLoopWords > maxCommonWords)
            maxCommonWords = (*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;
    int nscores = 0;

    /// 3.遍历所有闭环候选帧，计算相似度分数,保留那些分数高于minscore的帧,存入lScoreAndMatch
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                 lend = lKFsSharingWords.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        /// pKF只和具有共同单词较多的关键帧进行比较，需要大于minCommonWords
        if (pKFi->mnLoopWords > minCommonWords) {
            nscores++; // 这个变量后面没有用到
            /// 计算相似度评分
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    /// 如果没有超过指定相似度阈值的，那么就返回空
    if (lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*> > lAccScoreAndMatch; // acc:accumulate
    float bestAccScore = minScore;

    /// 对lScoreAndMatch中每一个KeyFrame都把与自己共视程度较高的帧归为一组,
    /// 每一组会计算组得分并记录该组分数最高的KeyFrame，记录于lAccScoreAndMatch
    for (list<pair<float, KeyFrame*> >::iterator it = lScoreAndMatch.begin(),
                 itend = lScoreAndMatch.end(); it != itend; it++) {
        KeyFrame* pKFi = it->second;
        /// 返回共视图中与此keyframe连接的权值前10的节点keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first; // 该组最高分数
        float accScore = it->first;  // 该组累计得分
        KeyFrame* pBestKF = pKFi;    // 该组最高分数对应的关键帧
        for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                     vend = vpNeighs.end(); vit != vend; vit++) {
            KeyFrame* pKF2 = *vit;
            if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords) {
                accScore += pKF2->mLoopScore;       // 因为pKF2->mnLoopQuery==pKF->mnId，所以只有pKF2也在闭环候选帧中，才能贡献分数
                if (pKF2->mLoopScore > bestScore) { // 统计得到组里分数最高的KeyFrame
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)// 记录所有组中组得分最高的组
            bestAccScore = accScore;
    }

    /// 返回所有关键帧中分数超过0.75*bestScore的计划
    float minScoreToRetain = 0.75f * bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    /// 得到组得分大于minScoreToRetain的组，得到组中分数最高的关键帧 0.75*bestScore
    for (list<pair<float, KeyFrame*> >::iterator it = lAccScoreAndMatch.begin(),
                 itend = lAccScoreAndMatch.end(); it != itend; it++) {
        if (it->first > minScoreToRetain) {
            KeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi)) { // 判断该pKFi是否已经在队列中了
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}

/**
 * @brief 在重定位中找到与该帧相似的关键帧
 *          1. 找出和当前帧具有公共单词的所有关键帧
 *          2. 只和具有共同单词较多的关键帧进行相似度计算
 *          3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
 *          4. 只返回累计得分较高的组中分数最高的关键帧
 * @param F 需要重定位的帧
 * @return  相似的关键帧
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame* F) {
    /// 相对于闭环检测DetectLoopCandidates，重定位检测中没法获得相连的关键帧
    list<KeyFrame*> lKFsSharingWords;
    /// 1.找出和当前帧具有公共单词的所有关键帧
    {
        unique_lock<mutex> lock(mMutex);
        /// 遍历该pKF的每一个word
        for (DBoW2::BowVector::const_iterator vit = F->mBowVec.begin(),
                     vend = F->mBowVec.end(); vit != vend; vit++) {
            /// 提取所有包含该word的KeyFrame
            list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];
            for (list<KeyFrame*>::iterator lit = lKFs.begin(),
                        lend = lKFs.end(); lit != lend; lit++) {
                KeyFrame* pKFi = *lit;
                if (pKFi->mnRelocQuery != F->mnId) { // pKFi还没有标记为pKF的候选帧
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    /// 2.统计所有闭环候选帧中与当前帧F具有共同单词最多的单词数，并以此决定阈值
    int maxCommonWords = 0;
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                 lend = lKFsSharingWords.end(); lit != lend; lit++) {
        if ((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;
    list<pair<float, KeyFrame*> > lScoreAndMatch;
    int nscores=0;

    /// 3. 遍历所有闭环候选帧，挑选出共有单词数大于阈值minCommonWords且单词匹配度大于minScore存入lScoreAndMatch
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                 lend = lKFsSharingWords.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        /// 当前帧F只和具有共同单词较多的关键帧进行比较，需要大于minCommonWords
        if (pKFi->mnRelocWords > minCommonWords) {
            nscores++;// 这个变量后面没有用到
            float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    /// 对lScoreAndMatch中每一个KeyFrame都把与自己共视程度较高的帧归为一组,
    /// 每一组会计算组得分并记录该组分数最高的KeyFrame，记录于lAccScoreAndMatch
    for (list<pair<float, KeyFrame*> >::iterator it = lScoreAndMatch.begin(),
                 itend = lScoreAndMatch.end(); it != itend; it++) {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        float bestScore = it->first; // 该组最高分数
        float accScore = bestScore;  // 该组累计得分
        KeyFrame* pBestKF = pKFi;    // 该组最高分数对应的关键帧
        for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                     vend = vpNeighs.end(); vit != vend; vit++) {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery != F->mnId)
                continue;
            accScore += pKF2->mRelocScore;          // 只有pKF2也在闭环候选帧中，才能贡献分数
            if (pKF2->mRelocScore > bestScore) {    // 统计得到组里分数最高的KeyFrame
                pBestKF = pKF2;
                bestScore = pKF2->mRelocScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore) // 记录所有组中组得分最高的组
            bestAccScore = accScore; // 得到所有组中最高的累计得分
    }

    /// 得到组得分大于阈值的，组内得分最高的关键帧
    float minScoreToRetain = 0.75f * bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, KeyFrame*> >::iterator it = lAccScoreAndMatch.begin(),
                 itend = lAccScoreAndMatch.end(); it != itend; it++) {
        const float& si = it->first;
        /// 只返回累计得分大于minScoreToRetain的组中分数最高的关键帧 0.75*bestScore
        if (si > minScoreToRetain) {
            KeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi)) { // 判断该pKFi是否已经在队列中了
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
