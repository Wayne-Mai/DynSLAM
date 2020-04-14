#include "../ImageLogReader.h"
#include "Parse.h"

int main(int argc, char* argv[]){
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

    float confGlobalInit, confObjectInit, icpErrThresh, covThresh, photoThresh,
        fernThresh;

    int timeDelta, icpCountThresh, start, end, preallocatedModelsCount,
        frameQueueSize;

    bool fillIn, openLoop, reloc, frameskip, quit, fastOdom, so3, rewind,
        frameToFrameRGB, usePrecomputedMasksOnly;

    int framesToSkip;
    bool streaming;
    bool resetButton;

    std::string tmpString;
    float tmpFloat;

    // get args

    iclnuim = Parse::get().arg(argc, argv, "-icl", tmpString) > -1;

    std::string baseDir;
    Parse::get().arg(argc, argv, "-basedir", baseDir);
    if (baseDir.length()) baseDir += '/';

    std::string calibrationFile;
    Parse::get().arg(argc, argv, "-cal", calibrationFile);
    if (calibrationFile.size()) calibrationFile = baseDir + calibrationFile;

    // Asus is default camera (might change later)
    if (Parse::get().arg(argc, argv, "-v2", tmpString) > -1) {
        Resolution::setResolution(512, 424);
        Intrinsics::setIntrinics(528, 528, 256, 212);
    } else if (Parse::get().arg(argc, argv, "-tum3", tmpString) > -1) {
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
    //              " cy: " << Intrinsics::getInstance().cy() << "]" <<
    //              std::endl;

    bool logReaderReady = false;

    Parse::get().arg(argc, argv, "-l", logFile);
    if (logFile.length()) {
        if (boost::filesystem::exists(logFile) &&
            boost::algorithm::ends_with(logFile, ".klg")) {
            logReader = std::make_unique<KlgLogReader>(
                logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        } else {
            logReader = std::make_unique<PangolinReader>(
                logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
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
            // else depthDir = logFile;
            if (maskDir.length()) maskDir += '/';
            // else maskDir = logFile;
            int indexW = -1;
            ImageLogReader* imageLogReader = new ImageLogReader(
                baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
                Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW
                                                                     : 4,
                colorPrefix, depthPrefix, maskPrefix,
                Parse::get().arg(argc, argv, "-f", tmpString) > -1);

            // How many masks?
            int maxMasks = -1;
            if (Parse::get().arg(argc, argv, "-nm", maxMasks) > -1) {
                if (maxMasks >= 0)
                    imageLogReader->setMaxMasks(maxMasks);
                else {
                    imageLogReader->ignoreMask();
#ifdef WAYNE_DEBUG
                    std::cout
                        << "\n Log : Image LogReader Ready in Ignore Mask.\n";
#endif
                }
            }

            logReader = std::unique_ptr<LogReader>(imageLogReader);
            usePrecomputedMasksOnly = imageLogReader->hasPrecomputedMasksOnly();
            logReaderReady = true;

#ifdef WAYNE_DEBUG
            std::cout << "\n Log : Use precomputed mask ? : "
                      << usePrecomputedMasksOnly << std::endl;
            ;
#endif
        }
    }





}