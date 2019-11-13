#include "stdafx.h"
#include "OpenCV.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
//RNG rng(12345);

int _tmain(int argc, _TCHAR* argv[])
{	
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	VideoCapture cap(0);
	if(!cap.isOpened())
		return -1;

	Mat frame;

	while(true)
	{
		//-- 3. Apply the classifier to the frame
		cap >> frame;

		if (!frame.empty())
		{
			detectAndDisplay(frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}
		
		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.3, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		char msg[512];
		
		float p = faces[i].area();
		float w1 = 4.9244, w2 = -7.0632, b1 = 6.1697, b2 = 6.2337;
		float norm_p, perkalian1, jumlah1, sigmoid1, perkalian2, jumlah2, linear2, denorm_y;
		float min_p = 2900, max_p = 63504;
		norm_p = 2 * (p - min_p) / (max_p - min_p) - 1;
		perkalian1 = 0;
		perkalian1 += w1*norm_p;
		jumlah1 = perkalian1 + b1;
		sigmoid1 = 1 / (1 + exp(-jumlah1));
		perkalian2 = 0;
		perkalian2 += w2 * sigmoid1;
		jumlah2 = perkalian2 + b2;
		float min_y = 30, max_y = 180;
		denorm_y = (jumlah2 + 1)*(max_y - min_y) / 2 + min_y;

		sprintf_s(msg, "F: %1f cm", denorm_y);
		printf("%s \n", msg);
		putText(frame, msg, Point(20, 30), FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, Scalar(0, 0, 255), 2);
	}

	//-- Show what you got
	imshow(window_name, frame);
}
