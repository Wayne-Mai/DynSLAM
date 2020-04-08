#include "../Core/Utils/Macros.h"
#include "MainController.h"
#include "Tools/KlgLogReader.h"
#include "Tools/OpenNI2LiveReader.h"
#ifdef WITH_FREENECT2
#include "Tools/FreenectLiveReader.h"
#endif
#include "Tools/ImageLogReader.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <GUI/Tools/PangolinReader.h>
#include <opencv2/core/ocl.hpp>
#include <toml.hpp>

#ifdef WAYNE_DEBUG
    int WRITE_MASK_INDEX = 0;
    const std::string WRITE_MASK_DIR = "/data/wayne/SLAM/train_log/";
#endif

#include "../Core/Segmentation/MaskRCNN/MaskRCNN.h"

class MaskRCNN;

int main(int argc, char* argv[]){


// mainControler staff
  bool good;
  MaskFusion* maskFusion;
  GUI* gui;
  bool showcaseMode;
  GroundTruthOdometry* groundTruthOdometry;
  std::unique_ptr<LogReader> logReader;

  bool iclnuim;
  std::string logFile;
  std::string poseFile;
  std::string exportDir;
  bool exportSegmentation;
  bool exportViewport;
  bool exportLabels;
  bool exportNormals;
  bool exportPoses;
  bool exportModels;

  float confGlobalInit, confObjectInit, icpErrThresh, covThresh, photoThresh, fernThresh;

  int timeDelta, icpCountThresh, start, end, preallocatedModelsCount, frameQueueSize;

  bool fillIn, openLoop, reloc, frameskip, quit, fastOdom, so3, rewind, frameToFrameRGB, usePrecomputedMasksOnly;

  int framesToSkip;
  bool streaming;
  bool resetButton;
  Segmentation::Method segmentationMethod;

  GPUResize* resizeStream;

  std::set<int> trackableClassIds;



    std::string tmpString;
    float tmpFloat;

    iclnuim = Parse::get().arg(argc, argv, "-icl", tmpString) > -1;

    std::string baseDir;
    Parse::get().arg(argc, argv, "-basedir", baseDir);
    if (baseDir.length()) baseDir += '/';

    std::string calibrationFile;
    Parse::get().arg(argc, argv, "-cal", calibrationFile);
    if (calibrationFile.size()) calibrationFile = baseDir + calibrationFile;

    // Asus is default camera (might change later)
    if(Parse::get().arg(argc, argv, "-v2", tmpString) > -1){
        Resolution::setResolution(512, 424);
        Intrinsics::setIntrinics(528, 528, 256, 212);
    } else if(Parse::get().arg(argc, argv, "-tum3", tmpString) > -1) {
        Resolution::setResolution(640, 480);
        Intrinsics::setIntrinics(535.4, 539.2, 320.1, 247.6);
    } else {
        Resolution::setResolution(640, 480);
        Intrinsics::setIntrinics(528, 528, 320, 240);
    }

    // if (calibrationFile.length()) loadCalibration(calibrationFile);
    // std::cout << "Calibration set to resolution: " <<
    //              Resolution::getInstance().width() << "x" <<
    //              Resolution::getInstance().height() <<
    //              ", [fx: " << Intrinsics::getInstance().fx() <<
    //              " fy: " << Intrinsics::getInstance().fy() <<
    //              ", cx: " << Intrinsics::getInstance().cx() <<
    //              " cy: " << Intrinsics::getInstance().cy() << "]" << std::endl;

    bool logReaderReady = false;

    Parse::get().arg(argc, argv, "-l", logFile);
    if (logFile.length()) {
        if (boost::filesystem::exists(logFile) && boost::algorithm::ends_with(logFile, ".klg")) {
            logReader = std::make_unique<KlgLogReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        } else {
            logReader = std::make_unique<PangolinReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        }
        logReaderReady = true;
    }

    if (!logReaderReady) {
        Parse::get().arg(argc, argv, "-dir", logFile);
        if (logFile.length()) {
            logFile += '/';  // "colorDir"
            std::string depthDir, maskDir, depthPrefix, colorPrefix, maskPrefix;
            Parse::get().arg(argc, argv, "-depthdir", depthDir);
            Parse::get().arg(argc, argv, "-maskdir", maskDir);
            Parse::get().arg(argc, argv, "-colorprefix", colorPrefix);
            Parse::get().arg(argc, argv, "-depthprefix", depthPrefix);
            Parse::get().arg(argc, argv, "-maskprefix", maskPrefix);
            if (depthDir.length()) depthDir += '/';
            //else depthDir = logFile;
            if (maskDir.length()) maskDir += '/';
            //else maskDir = logFile;
            int indexW = -1;
            ImageLogReader* imageLogReader = new ImageLogReader(baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
                                                                Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW : 4, colorPrefix,
                                                                depthPrefix, maskPrefix, Parse::get().arg(argc, argv, "-f", tmpString) > -1);

            // How many masks?
            int maxMasks = -1;
            if (Parse::get().arg(argc, argv, "-nm", maxMasks) > -1) {
                if (maxMasks >= 0)
                    imageLogReader->setMaxMasks(maxMasks);
                else{
                    imageLogReader->ignoreMask();
                    # ifdef WAYNE_DEBUG
                    std::cout<<"\n Log : Image LogReader Ready in Ignore Mask.\n";
                    # endif

                }
            }

            logReader = std::unique_ptr<LogReader>(imageLogReader);
            usePrecomputedMasksOnly = imageLogReader->hasPrecomputedMasksOnly();
            logReaderReady = true;

            # ifdef WAYNE_DEBUG
            std::cout<<"\n Log : Use precomputed mask ? : "<<usePrecomputedMasksOnly<<std::endl;;
            # endif

        }
    }



# ifdef WAYNE_DEBUG
    std::cout<<"\n Log : Image-loger initialize success.\n";
# endif


    // Start process frame
    FrameDataPointer frame=logReader->getFrameData();

    // note initialize MaskRCNN module
    // std::unique_ptr<MaskRCNN> maskRCNN;
    // bool sequentialMaskRCNN;
    // std::queue<FrameDataPointer> queue; //  should be nullptr ???
    bool sequentialMaskRCNN=true;
    std::unique_ptr<MaskRCNN> maskRCNN;

    // // if(embedMaskRCNN){ // usePreComputedMask
    if(true){
        maskRCNN = std::make_unique<MaskRCNN>(nullptr); // queue in sourceCode
        // sequentialMaskRCNN = (queue == nullptr);
    }


    // if (maskFusion->processFrame(logReader->getFrameData(), currentPose, weightMultiplier) && !showcaseMode) {
    //                 gui->pause->Ref().Set(true);
    //             }
    // logReader->getFrameData == frame.

    // SegmentationResult segmentationResult = performSegmentation(frame);

    // SegmentationResult MaskFusion::performSegmentation(FrameDataPointer frame) {
    //     return labelGenerator.performSegmentation(models, frame, getNextModelID(), spawnOffset >= modelSpawnOffset);
    // }

    // SegmentationResult MfSegmentation::performSegmentation(std::list<std::shared_ptr<Model> > &models,
    //                                                        FrameDataPointer frame,
    //                                                        unsigned char nextModelID,
    //                                                        bool allowNew){


    if(frame->mask.total() == 0) {
    if(!maskRCNN) throw std::runtime_error("MaskRCNN is not embedded and no masks were pre-computed.");
    else if(sequentialMaskRCNN) maskRCNN->executeSequential(frame);
    }

#ifdef WAYNE_DEBUG
    cv::imwrite(WRITE_MASK_DIR + "mrcnn" + std::to_string(frame->index) + ".png", frame->mask);
#endif



    return 0;


    // FrameDataPointer result = std::make_shared<FrameData>();

    // ImageLogReader* imageLogReader = new ImageLogReader(baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
    //                                                 Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW : 4, colorPrefix,
    //                                                 depthPrefix, maskPrefix, Parse::get().arg(argc, argv, "-f", tmpString) > -1);

}


// FrameDataPointer ImageLogReader::loadFrameFromDrive(const size_t& index) {
//   FrameDataPointer result = std::make_shared<FrameData>();

//   // Get path to image files
//   std::stringstream ss;
//   ss << std::setw(indexW) << std::setfill('0') << index + startIndex;
//   std::string indexStr = ss.str();

//   std::string depthImagePath = depthImagesDir + depthPre + indexStr + depthExt;
//   if (!boost::filesystem::exists(depthImagePath)) throw std::invalid_argument("Could not find depth-image file: " + depthImagePath);

//   std::string rgbImagePath = file + colorPre + indexStr + colorExt;
//   if (!boost::filesystem::exists(rgbImagePath)) throw std::invalid_argument("Could not find rgb-image file: " + rgbImagePath);

//   // Load mask ids
//   //std::string maskImagePath = maskImagesDir + maskPre + std::to_string(index);
//   std::string maskImagePath = maskImagesDir + maskPre + indexStr;
//   std::string maskDescrPath = maskImagePath + ".txt";
//   maskImagePath += maskExt;
//   if (hasMasksGT) {
//     if (!boost::filesystem::exists(maskImagePath)) throw std::invalid_argument("Could not find mask-image file: " + maskImagePath);
//     if (boost::filesystem::exists(maskDescrPath)) loadMaskIDs(maskDescrPath, &result->classIDs, &result->rois);
//   }


//   // Load RGB
//   result->rgb = cv::imread(rgbImagePath);
//   if (result->rgb.total() == 0) throw std::invalid_argument("Could not read rgb-image file.");
//   result->flipColors();

//   // Load Depth
//   result->depth = cv::imread(depthImagePath, cv::IMREAD_UNCHANGED);
//   if (result->depth.total() == 0) throw std::invalid_argument("Could not read depth-image file. (Empty)");
//   if (result->depth.type() != CV_32FC1) {
//     cv::Mat newDepth(result->depth.rows, result->depth.cols, CV_32FC1);
//     if (result->depth.type() == CV_32FC3) {
//       unsigned depthIdx = 0;
//       for (int i = 0; i < result->depth.rows; ++i) {
//         cv::Vec3f* pixel = result->depth.ptr<cv::Vec3f>(i);
//         for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = pixel[j][0];
//       }

//     } else if (result->depth.type() == CV_16UC1) {
//       //std::cout << "Warning -- your depth scale is likely to mismatch. Check ImageLogReader.cpp!" << std::endl;
//       unsigned depthIdx = 0;
//       for (int i = 0; i < result->depth.rows; ++i) {
//         unsigned short* pixel = result->depth.ptr<unsigned short>(i);
//         for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = 0.001f * pixel[j];
//         //for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = 0.0002f * pixel[j]; // FIXME
//       }
//     } else {
//       throw std::invalid_argument("Unsupported depth-files: " + cvTypeToString(result->depth.type()));
//     }
//     result->depth = newDepth;
//   }

//   // Load Mask
//   if (hasMasksGT && (index < maxMasks)) {
//     result->mask = cv::imread(maskImagePath, cv::IMREAD_GRAYSCALE);
//     if (result->mask.total() != result->rgb.total()) throw std::invalid_argument("Could not read mask-image file.");
//     if (!result->mask.isContinuous() || result->mask.type() != CV_8UC1) throw std::invalid_argument("Incompatible mask image.");
//   }

//   result->timestamp = index * 1000.0f / rateHz;
//   result->index = index;

//   return result;
// }



// void ImageLogReader::bufferFramesImpl() {
//   if (int(nextBufferIndex) - minBuffered + 1 > int(currentFrame)) return;
//   for (unsigned i = 0; i < 15 && nextBufferIndex < frames.size(); ++i, ++nextBufferIndex) {
//     frames[nextBufferIndex] = loadFrameFromDrive(nextBufferIndex);
//   }
// }


//   // Data
//   std::vector<FrameDataPointer> frames;


//   FrameDataPointer ImageLogReader::getFrameData() {
//   // assert(frames[currentFrame] != 0);
//   if (currentFrame < 0) return NULL;

//   bool bufferFail = false;
//   while (!frames[currentFrame] || frames[currentFrame]->depth.total() == 0) {
//     usleep(1);
//     bufferFail = true;
//   }
//   if (bufferFail) std::cout << "Buffering failure." << std::endl;
//   return frames[currentFrame];
//  }


  
//    FrameDataPointer getFrameData();
  
//    imageLogReader* imageLogReader = new ImageLogReader(baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
//                                                                 Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW : 4, colorPrefix,
//                                                                 depthPrefix, maskPrefix, Parse::get().arg(argc, argv, "-f", tmpString) > -1);

//     // How many masks?
//     int maxMasks = -1;
//     if (Parse::get().arg(argc, argv, "-nm", maxMasks) > -1) {
//         if (maxMasks >= 0)
//             imageLogReader->setMaxMasks(maxMasks);
//         else
//             imageLogReader->ignoreMask();
//     }

//     logReader = std::unique_ptr<LogReader>(imageLogReader);
   
   
//    // note get frame data
//     if (maskFusion->processFrame(logReader->getFrameData(), currentPose, weightMultiplier) && !showcaseMode) {
//         gui->pause->Ref().Set(true);
//     }

//     bool MaskFusion::processFrame(FrameDataPointer frame, const Eigen::Matrix4f* inPose, const float weightMultiplier, const bool bootstrap) {


//     assert(frame->depth.type() == CV_32FC1);
//     assert(frame->rgb.type() == CV_8UC3);
//     assert(frame->timestamp >= 0);
//     TICK("Run");

//     frameQueue.push(frame);
//     if(frameQueue.size() < queueLength) return 0;
//     frame = frameQueue.front();
//     frameQueue.pop();

//     // Upload RGB to graphics card
//     textureRGB->texture->Upload(frame->rgb.data, GL_RGB, GL_UNSIGNED_BYTE);


//     // note maskfusion class preprocess
//     TICK("Preprocess");

//     textureDepthMetric->texture->Upload((float*)frame->depth.data, GL_LUMINANCE, GL_FLOAT);
//     filterDepth();

//     // if(frame.mask) {
//     //    // Use ground-truth segmentation if provided (TODO: Overwritten at the moment)
//     //    textureMask->texture->Upload(frame.mask, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
//     //} else
//     if (!enableMultipleModels) {
//         // If the support for multiple objects is deactivated, segment everything as background (static scene).
//         const long size = Resolution::getInstance().width() * Resolution::getInstance().height();
//         unsigned char* data = new unsigned char[size];
//         memset(data, 0, size);
//         textureMask->texture->Upload(data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
//         delete[] data;
//     }

//     TOCK("Preprocess");


//    // note maskfusion call perform segmentation function
//     SegmentationResult segmentationResult = performSegmentation(frame);


//    // create MaskFusion class
//    if(embedMaskRCNN){
//         maskRCNN = std::make_unique<MaskRCNN>(queue);
//         sequentialMaskRCNN = (queue == nullptr);
//     }
   
   
//     // TODO does MaskRCNN finished here ?? YES. result in frame-mask
//     if(frame->mask.total() == 0) {
//         if(!maskRCNN) throw std::runtime_error("MaskRCNN is not embedded and no masks were pre-computed.");
//         else if(sequentialMaskRCNN) maskRCNN->executeSequential(frame);
//     }


