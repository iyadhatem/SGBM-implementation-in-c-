#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>


#define CVUI_IMPLEMENTATION
#include "cvui.h"

// Include the class that provides an enhanced cvui window component
#include "EnhancedWindow.h"

#define WINDOW_NAME	"CVUI Ehnanced UI SGBM"


// initialize values for StereoSGBM parameters
int numDisparities = 1;
int blockSize = 1;
int preFilterType = 1;
int preFilterSize = 0;
int preFilterCap = 0;
int minDisparity = 0;
int uniquenessRatio = 0;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = 0;
int dispType = CV_16S;
int P1=0,P2=0;

// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();


cv::Mat disparity;





int main()
{
    //-- 1. Read the images
    std::string imSource = "intenta";
    std::string img1_filename = "../images/"+imSource+"/view1.png";
    std::string img2_filename = "../images/"+imSource+"/view5.png";

    cv::Mat imgL;
    cv::Mat imgR;

    imgL = cv::imread(img1_filename);
    imgR = cv::imread(img2_filename);
    int w,h;
    h=imgL.rows;
    w=imgL.cols;
    int n=3;
    int cn = imgL.channels();

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4, STEREO_HH4=5 };
    int alg = STEREO_SGBM;

    if(alg==STEREO_HH)
        stereo->setMode(cv::StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        stereo->setMode(cv::StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_HH4)
        stereo->setMode(cv::StereoSGBM::MODE_HH4);
    else if(alg==STEREO_3WAY)
        stereo->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disparity16S, disparity8U,disparity32F;;
// Create a settings window using the EnhancedWindow class.
    EnhancedWindow settings(10, 50, 270, 570,  "Settings");
    // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
    cvui::init(WINDOW_NAME);
    cv::resizeWindow(WINDOW_NAME,w*n,h*n);// increase image size for display purposes

    cv::Mat imgL_gray,  imgR_gray;
    while (true)
    {
        // Converting images to grayscale
        cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);

        stereo->setBlockSize(blockSize*2+1);
        stereo->setP1(P1*8*cn*blockSize*blockSize);
        stereo->setP2(P2*32*cn*blockSize*blockSize);
        stereo->setMinDisparity(minDisparity);
        stereo->setNumDisparities(numDisparities*16);
        stereo->setPreFilterCap(preFilterCap);
        stereo->setUniquenessRatio(uniquenessRatio);
        stereo->setSpeckleRange(speckleRange);
        stereo->setSpeckleWindowSize(speckleWindowSize);
        stereo->setDisp12MaxDiff(disp12MaxDiff);

        // Calculating disparith using the StereoBM algorithm
        stereo->compute(imgL_gray,imgR_gray,disparity16S);

        //     cv::ximgproc::getDisparityVis(disparity16S,disparity8U,15);
       // imwrite("../viewImg/bin/Debug/disparityImg.tif", disparity16S); // save it for pixel value (disparity) detection
        imwrite("disparityImg.tif", disparity16S); // save it for pixel value (disparity) detection


        // NOTE: Code returns a 16bit signed single channel image,
        // CV_16S containing a disparity map scaled by 16. Hence it
        // is essential to convert it to CV_32F and scale it down 16 times.



        // Displaying the disparity map
        cv::Mat disparity8U_3c;

        double minVal, maxVal;
        minMaxLoc(disparity16S, &minVal, &maxVal);
        disparity16S.convertTo(disparity8U, CV_8UC1, 255/(maxVal - minVal));
        cv::applyColorMap(disparity8U, disparity8U_3c, cv::COLORMAP_JET);//{TURBO  PARULA  JET

        cv::Mat disparity = disparity8U_3c.clone();
        cv::resize(disparity, disparity, cv::Size(w*n, h*n), cv::INTER_LINEAR);
        // Render the settings window and its content, if it is not minimized.
        settings.begin(disparity);
        if (!settings.isMinimized())
        {
            cvui::printf("      numDisparities");
            cvui::trackbar(settings.width() - 20, &numDisparities, 1, 255/16);
            cvui::printf("      blockSize");
            cvui::trackbar(settings.width() - 20, &blockSize,  1, 5);
            cvui::printf("      minDisparity");
            cvui::trackbar(settings.width() - 20, &minDisparity,  -100, 255);
            cvui::printf("      preFilterCap");
            cvui::trackbar(settings.width() - 20, &preFilterCap, 0, 100);
            cvui::printf("      uniquenessRatio");
            cvui::trackbar(settings.width() - 20, &uniquenessRatio,  5, 15);
            cvui::printf("      speckleRange");
            cvui::trackbar(settings.width() - 20, &speckleRange,  1, 2);
            cvui::printf("      speckleWindowSize");
            cvui::trackbar(settings.width() - 20, &speckleWindowSize, 50, 200);
            cvui::printf("      disp12MaxDiff");
            cvui::trackbar(settings.width() - 20, &disp12MaxDiff,  -1, 100);
            cvui::printf("          P1");
            cvui::trackbar(settings.width() - 20, &P1, 0, 1);
            cvui::printf("          P2");
            cvui::trackbar(settings.width() - 20, &P2,  0, 1);

        }
        settings.end();

        cv::imshow(WINDOW_NAME,disparity);

        // Close window using esc key
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
