#include "pch.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
	vector<Mat> Myimg;
	vector<KeyPoint> Keypoint1, Keypoint2;
	vector<DMatch> matches1, matches2;
	vector<Point2f> point1, point2;
	Mat img1, img2, descriptor1, descriptor2;
	Mat HomoMat, Panorama, out;
	Mat image = imread("left.jpg");
	Myimg.push_back(image);
	
	image = imread("right.jpg");
	Myimg.push_back(image);

	cvtColor(Myimg[0], img1, COLOR_BGRA2GRAY);
	cvtColor(Myimg[1], img2, COLOR_BGRA2GRAY);

	Ptr<FeatureDetector> detector = BRISK::create(90);

	detector->detectAndCompute(img1, Mat(), Keypoint1, descriptor1);
	detector->detectAndCompute(img2, Mat(), Keypoint2, descriptor2);

	Ptr<BFMatcher> Mather = BFMatcher::create(NORM_HAMMING);
	Mather->match(descriptor1, descriptor2, matches1);

	for (auto& m : matches1) {
		if (m.distance < 20) {
			matches2.push_back(m);
		}
	}

	for (int i = 0; i < matches2.size(); i++) {
		point1.push_back(Keypoint1[matches2[i].queryIdx].pt);
		point2.push_back(Keypoint2[matches2[i].trainIdx].pt);
	}

	HomoMat = findHomography(point2, point1, RANSAC);

	warpPerspective(Myimg[1], Panorama, HomoMat, Size(Myimg[1].cols * 2, Myimg[1].rows), INTER_CUBIC);

	Mat ROI(Panorama, Rect(0, 0, Myimg[0].cols, Myimg[0].rows));
	Myimg[0].copyTo(ROI);

	imshow("out", Panorama);
	waitKey(0);
}
