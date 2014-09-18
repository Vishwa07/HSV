// ConsoleApplication1.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include <string>
#include "windows.h"
#include <cstring>

/**
* @function calcHist_Demo.cpp
* @brief Demo code to use the function calcHist
* @author
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
double func(string img1,string img2)
{
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1;
	

	

	src_base = imread(img1,1);
	src_test1 = imread(img2,1);
	

	/// Convert to HSV
	cvtColor(src_base, hsv_base, CV_BGR2HSV);
	cvtColor(src_test1, hsv_test1, CV_BGR2HSV);


	

	/// Using 30 bins for hue and 32 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 256, saturation from 0 to 180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	/// Histograms
	MatND hist_base;
	MatND hist_half_down;
	MatND hist_test1;
	MatND hist_test2;

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());	

    calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	/// Apply the histogram comparison methods
	
	
		
		double base_test1 = compareHist(hist_base, hist_test1, 1);
		return  base_test1;

	


}
double func1(string img1,string img2)
{
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1;
	src_base = imread(img1, 1);
	src_test1 = imread(img2,1);
	//cvtColor(src_base, src_test1, CV_RGB2GRAY);
	cv::inRange(src_base, cv::Scalar(40, 40, 40),cv::Scalar(250, 250, 250),src_base);
	cv::inRange(src_test1, cv::Scalar(40, 40, 40),cv::Scalar(250, 250, 250),src_test1);
	cv::bitwise_not(src_base, src_base); 
	cv::bitwise_not(src_test1, src_test1); 
	 cv::initModule_nonfree() ;
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
  vector<KeyPoint> keypoints;
  vector<KeyPoint> keypoints1;

  // Detect the keypoints
  featureDetector->detect(src_base, keypoints); // NOTE: featureDetector is a pointer hence the '->'.
 featureDetector->detect(src_test1, keypoints1);
  //Similarly, we create a smart pointer to the SIFT extractor.
  Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");

  // Compute the 128 dimension SIFT descriptor at each keypoint.
  // Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
  Mat descriptors,descriptors1;
  featureExtractor->compute(src_base, keypoints, descriptors);
  featureExtractor->compute(src_base, keypoints1, descriptors1);

  //BruteForceMatcher<L2<float> > matcher;
  BFMatcher matcher(NORM_L2);
vector<DMatch> matches;
matcher.match(descriptors, descriptors1, matches);

 Mat img_matches;
 drawMatches( src_base, keypoints, src_test1, keypoints1, matches, img_matches );

  //-- Show detected matches
  imshow("Matches", img_matches );
  //// If you would like to draw the detected keypoint just to check
  //Mat outputImage;
  //Scalar keypointColor = Scalar(255, 255, 255);     // Blue keypoints.
  //drawKeypoints(src_base, keypoints, src_base, keypointColor, DrawMatchesFlags::DEFAULT);


  
	//char *Win_name = "asdasd";
	//namedWindow(Win_name);
	////imwrite("F:\\Datamonster\\ImageBasedProductMatching\\data\\a.jpg",src_base);
	//imshow(Win_name,src_base);
		waitKey(0);
	return 0.0;
}
double sobel1(string img1,string img2)
{
Mat src_test1, src_test2;
	
	Mat src_gray;
	Mat grad;
	//char* window_name = "Sobel Demo - Simple Edge Detector";
	//char* window_name1 = "Sobel Demo1 - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;




	src_test1 = imread(img1, 1);
	src_test2 = imread(img2, 1);

	GaussianBlur(src_test1, src_test1, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_test1, src_gray, CV_RGB2GRAY);

	/// Create window
	//namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//imshow(window_name, grad);

	//waitKey(0);

	Mat src_gray2;
	Mat grad2;

	GaussianBlur(src_test2, src_test2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_test2, src_gray2, CV_RGB2GRAY);

	/// Create window
	//namedWindow(window_name1, CV_WINDOW_AUTOSIZE);

	/// Generate grad_x and grad_y
	Mat grad_x2, grad_y2;
	Mat abs_grad_x2, abs_grad_y2;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray2, grad_x2, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x2, abs_grad_x2);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray2, grad_y2, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y2, abs_grad_y2);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, grad2);

	//imshow(window_name1, grad2);
	int threshold = (double)(grad.rows * grad.cols) * 0.70; 
	Mat result;
	cv::compare(grad , grad2  , result , cv::CMP_EQ );
	int similarPixels  = countNonZero(result);

	if ( similarPixels  > threshold ) {
  // cout << "similar" << endl;
   return 1.0;
	}
	return 0.0;
}
double sobelOperator(string img1, string img2)
{
	
	Mat src_test1, src_test2;
	
	Mat src_gray;
	Mat grad;
	//char* window_name = "Sobel Demo - Simple Edge Detector";
	//char* window_name1 = "Sobel Demo1 - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;




	src_test1 = imread(img1, 1);
	src_test2 = imread(img2, 1);

	GaussianBlur(src_test1, src_test1, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_test1, src_gray, CV_RGB2GRAY);

	/// Create window
	//namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//imshow(window_name, grad);

//	waitKey(0);

	Mat src_gray2;
	Mat grad2;

	GaussianBlur(src_test2, src_test2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_test2, src_gray2, CV_RGB2GRAY);

	/// Create window
//	namedWindow(window_name1, CV_WINDOW_AUTOSIZE);

	/// Generate grad_x and grad_y
	Mat grad_x2, grad_y2;
	Mat abs_grad_x2, abs_grad_y2;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray2, grad_x2, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x2, abs_grad_x2);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray2, grad_y2, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y2, abs_grad_y2);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, grad2);

	//imshow(window_name1, grad2);

	Mat hist1,hist2;

	int channels[] = { 0 };
	int histSize[] = { 32 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };

	calcHist(&grad, 1, channels, Mat(), // do not use mask
		hist1, 1, histSize, ranges,
		true, // the histogram is uniform
		false);

	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&grad2, 1, channels, Mat(), // do not use mask
		hist2, 1, histSize, ranges,
		true, // the histogram is uniform
		false);

	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

	double base_base = compareHist(hist1, hist2, 2);

	
		double base_test1 = compareHist(hist1, hist2, 1);
		//double base_test2 = compareHist(hist_base, hist_test2, compare_method);

	

	return base_test1;



}
int main(int, char** argv)
{
//double a = sobel("F:\\Datamonster\\ImageBasedProductMatching\\data\\6-2.jpg", "F:\\Datamonster\\ImageBasedProductMatching\\data\\6-8.jpg");

WIN32_FIND_DATA FindFileData;
   HANDLE hFind;

   hFind=FindFirstFile(L"F:\\Datamonster\\ImageBasedProductMatching\\data\\*.jpg",&FindFileData);
   Vector<string> v; 
		
   while (hFind != INVALID_HANDLE_VALUE) 
   {
	   char p[256];
	   wcstombs(p,FindFileData.cFileName,256);
	   String a(p);
		v.push_back(a);	  
	  bool istrue=FindNextFile(hFind,&FindFileData);
	  if(istrue==false)
		  break;
	}

   for(int i=0;i<v.size();i++)
   
	  {
		  double a=func("F:\\Datamonster\\ImageBasedProductMatching\\data\\4-0.jpg","F:\\Datamonster\\ImageBasedProductMatching\\data\\"+v[i]);
      if(a<=2.0)
  {
	 
	  imwrite("F:\\Datamonster\\ImageBasedProductMatching\\data1\\"+v[i],imread("F:\\Datamonster\\ImageBasedProductMatching\\data\\"+v[i]));
   }
   }
	     hFind=FindFirstFile(L"F:\\Datamonster\\ImageBasedProductMatching\\data1\\*.jpg",&FindFileData);
  v.clear(); 
		
   while (hFind != INVALID_HANDLE_VALUE) 
   {
	   char p[256];
	   wcstombs(p,FindFileData.cFileName,256);
	   String a(p);
		v.push_back(a);	  
	  bool istrue=FindNextFile(hFind,&FindFileData);
	  if(istrue==false)
		  break;
	}

   for(int i=0;i<v.size();i++)
   
	  {
		  double a=sobel1("F:\\Datamonster\\ImageBasedProductMatching\\data1\\4-0.jpg","F:\\Datamonster\\ImageBasedProductMatching\\data1\\"+v[i]);
			if(a==1.0)
			{
	 
				imwrite("F:\\Datamonster\\ImageBasedProductMatching\\data2\\"+v[i],imread("F:\\Datamonster\\ImageBasedProductMatching\\data1\\"+v[i]));
			}
	}
   cout<<"Images are dumped!!!";
}