#include "MaskFusion.h"

  
int main(){

    

    // note #2 use enum to create labelGenerator
    //          @1 only textureRGB and textureDepthMetric are inner-class 
    //          @2 so we create the whole MaskFusion class instead 
    
    // get args
    // Tmp variables for parameter parsing
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

    if (calibrationFile.length()) loadCalibration(calibrationFile);
    std::cout << "Calibration set to resolution: " <<
                 Resolution::getInstance().width() << "x" <<
                 Resolution::getInstance().height() <<
                 ", [fx: " << Intrinsics::getInstance().fx() <<
                 " fy: " << Intrinsics::getInstance().fy() <<
                 ", cx: " << Intrinsics::getInstance().cx() <<
                 " cy: " << Intrinsics::getInstance().cy() << "]" << std::endl;

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
                else
                    imageLogReader->ignoreMask();
            }

            logReader = std::unique_ptr<LogReader>(imageLogReader);
            usePrecomputedMasksOnly = imageLogReader->hasPrecomputedMasksOnly();
            logReaderReady = true;
        }
    }

    // Try live cameras

    // KinectV1 / Asus
    if (!logReaderReady && Parse::get().arg(argc, argv, "-v1", tmpString) > -1) {
        logReader = std::make_unique<OpenNI2LiveReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        good = ((OpenNI2LiveReader*)logReader.get())->asus->ok();
    }
    // KinectV2
    if(Parse::get().arg(argc, argv, "-v2", tmpString) > -1){
#ifdef WITH_FREENECT2
        assert(!logReaderReady);
        logReader = std::make_unique<FreenectLiveReader>();
        good = ((FreenectLiveReader*)logReader.get())->isDeviceGood();
#else
        throw std::invalid_argument("v2 support not enabled, set WITH_FREENECT2 during build.");
#endif
    }

    if(!logReader){
        std::cout << "No input data" << std::endl; // Todo: Try to find camera more automatically
        exit(1);
    }

    if (logReader->hasIntrinsics() && !calibrationFile.length()) loadCalibration(logReader->getIntinsicsFile());

    if (Parse::get().arg(argc, argv, "-p", poseFile) > 0) {
        groundTruthOdometry = new GroundTruthOdometry(poseFile);
    }

    showcaseMode = Parse::get().arg(argc, argv, "-sc", tmpString) > -1;
    gui = new GUI(logFile.length() == 0, showcaseMode);

    confObjectInit = 0.01f;
    confGlobalInit = 10.0f;
    //confGlobalInit = 0.01f;
    icpErrThresh = 5e-05f;
    covThresh = 1e-05f;
    photoThresh = 115;
    fernThresh = 0.3095f;
    preallocatedModelsCount = 0;
    frameQueueSize = 30;

    timeDelta = 200;  // Ignored, since openLoop
    icpCountThresh = 40000;
    start = 1;
    so3 = !(Parse::get().arg(argc, argv, "-nso", tmpString) > -1);
    end = std::numeric_limits<unsigned short>::max();  // Funny bound, since we predict times in this format really!

    Parse::get().arg(argc, argv, "-confG", confGlobalInit);
    Parse::get().arg(argc, argv, "-confO", confObjectInit);
    Parse::get().arg(argc, argv, "-ie", icpErrThresh);
    Parse::get().arg(argc, argv, "-cv", covThresh);
    Parse::get().arg(argc, argv, "-pt", photoThresh);
    Parse::get().arg(argc, argv, "-ft", fernThresh);
    Parse::get().arg(argc, argv, "-t", timeDelta);
    Parse::get().arg(argc, argv, "-ic", icpCountThresh);
    Parse::get().arg(argc, argv, "-s", start);
    Parse::get().arg(argc, argv, "-e", end);
    Parse::get().arg(argc, argv, "-a", preallocatedModelsCount);
    Parse::get().arg(argc, argv, "-frameQ", frameQueueSize);

    logReader->flipColors = Parse::get().arg(argc, argv, "-f", tmpString) > -1;

    openLoop = true;  // FIXME //!groundTruthOdometry && (Parse::get().arg(argc, argv, "-o", empty) > -1);
    reloc = Parse::get().arg(argc, argv, "-rl", tmpString) > -1;
    frameskip = Parse::get().arg(argc, argv, "-fs", tmpString) > -1;
    quit = Parse::get().arg(argc, argv, "-q", tmpString) > -1;
    fastOdom = Parse::get().arg(argc, argv, "-fo", tmpString) > -1;
    rewind = Parse::get().arg(argc, argv, "-r", tmpString) > -1;
    frameToFrameRGB = Parse::get().arg(argc, argv, "-ftf", tmpString) > -1;
    exportSegmentation = Parse::get().arg(argc, argv, "-es", tmpString) > -1;
    exportViewport = Parse::get().arg(argc, argv, "-ev", tmpString) > -1;
    exportLabels = Parse::get().arg(argc, argv, "-el", tmpString) > -1;
    exportNormals = Parse::get().arg(argc, argv, "-en", tmpString) > -1;
    exportPoses = Parse::get().arg(argc, argv, "-ep", tmpString) > -1;
    exportModels = Parse::get().arg(argc, argv, "-em", tmpString) > -1;
    Parse::get().arg(argc, argv, "-method", tmpString);
    if (tmpString == "cofusion") {
        segmentationMethod = Segmentation::Method::CO_FUSION;
        gui->addCRFParameter();
    } else {

        segmentationMethod = Segmentation::Method::MASK_FUSION;
        gui->addBifoldParameters();
    }

}  
  
