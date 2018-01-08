#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


int main()
{
	VideoCapture cap("MVI_1049.avi"); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	int _rows = 100;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	cvCreateTrackbar("Rows", "Control", &_rows, 1000);
	
	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // read a new frame from video
		resize(imgOriginal, imgOriginal, Size(imgOriginal.rows, _rows));

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		namedWindow("Original", CV_WINDOW_FREERATIO);
		imshow("Original", imgOriginal); //show the original image
		

		Mat imgHSV;
	

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HLS); //Convert the captured frame from BGR to HSV
	    // recover size of imgThresholded by size of imgOriginal
		//resize(imgOriginal, imgThresholded, Size(imgOriginal.rows, imgOriginal.cols));
	
		Mat imgThresholded;


		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

																									  //morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (fill small holes in the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));



		IplImage copy1 = imgThresholded; 
		IplImage* imgCanny = &copy1;// convert Mat to IplImage* 




		// Dò( tách) biên bằng phương pháp Canny
		IplImage* cannyImg = cvCreateImage(cvGetSize(imgCanny), imgCanny->depth, 1);
		cvCanny(imgCanny, cannyImg, 100, 200);

		////////////////


		
		Mat imgGray = cvarrToMat(cannyImg); // Convert IplImage to Mat
		


		// Reduce the noise so we avoid false circle detection
		GaussianBlur(imgGray, imgGray, Size(9, 9), 2, 2);

		vector<Vec3f> circles;

		// Apply the Hough Transform to find the circles
		
		HoughCircles(imgGray, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 50, 0, 0);
	
		/* Arguments:
		src_gray: Input image (grayscale).
		circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
		HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
		dp = 1: The inverse ratio of resolution.
		min_dist = 30: Minimum distance between detected centers.
		param_1 = 200: Upper threshold for the internal Canny edge detector.
		param_2 = 50*: Threshold for center detection.
		min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
		max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.
		*/

		// Draw the circles detected
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
		//	circle(imgOriginal, center, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center     
			//circle(imgOriginal, center, radius, Scalar(0, 0, 255), 3, 8, 0);// circle outline
			cout << "center : " << center << "\nradius : " << radius << endl;

			Rect crop = Rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
			Mat cropImg = imgOriginal(crop);
			namedWindow("cropoed", CV_WINDOW_FREERATIO);
			imshow("cropoed", cropImg);
			imwrite("D:/OpenCv/testDemo/img.jpg", cropImg);
			cout << cropImg.size();
			waitKey(0);
	

			
		}

		////////////////
		
	
		imshow("Thresholded Image", imgThresholded); //show the thresholded image
	

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;

}



// TrainAndTest.cpp
//
//#include<opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/ml/ml.hpp>
//
//#include<iostream>
//#include<sstream>
//
//// global variables ///////////////////////////////////////////////////////////////////////////////
//const int MIN_CONTOUR_AREA = 100;
//
//const int RESIZED_IMAGE_WIDTH = 20;
//const int RESIZED_IMAGE_HEIGHT = 30;
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//class ContourWithData {
//public:
//	// member variables ///////////////////////////////////////////////////////////////////////////
//	std::vector<cv::Point> ptContour;           // contour
//	cv::Rect boundingRect;                      // bounding rect for contour
//	float fltArea;                              // area of contour
//
//												///////////////////////////////////////////////////////////////////////////////////////////////
//	bool checkIfContourIsValid() {                              // obviously in a production grade program
//		if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
//		return true;                                            // identifying if a contour is valid !!
//	}
//
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
//		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
//	}
//
//};
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//int main() {
//	std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
//	std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly
//
//																// read in training classifications ///////////////////////////////////////////////////
//
//	cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector
//
//	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file
//
//	if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
//		std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
//		return(0);                                                                                  // and exit program
//	}
//
//	fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
//	fsClassifications.release();                                        // close the classifications file
//
//																		// read in training images ////////////////////////////////////////////////////////////
//
//	cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector
//
//	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file
//
//	if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
//		std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
//		return(0);                                                                              // and exit program
//	}
//
//	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
//	fsTrainingImages.release();                                                 // close the traning images file
//
//																				// train //////////////////////////////////////////////////////////////////////////////
//
//	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object
//
//																				// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
//																				// even though in reality they are multiple images / numbers
//	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);
//
//	// test ///////////////////////////////////////////////////////////////////////////////
//
//	cv::Mat matTestingNumbers = cv::imread("asd1.png");            // read in the test numbers image
//
//	if (matTestingNumbers.empty()) {                                // if unable to open image
//		std::cout << "error: image not read from file\n\n";         // show error message on command line
//		return(0);                                                  // and exit program
//	}
//
//	cv::Mat matGrayscale;           //
//	cv::Mat matBlurred;             // declare more image variables
//	cv::Mat matThresh;              //
//	cv::Mat matThreshCopy;          //
//
//	cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);         // convert to grayscale
//
//																		// blur
//	cv::GaussianBlur(matGrayscale,              // input image
//		matBlurred,                // output image
//		cv::Size(5, 5),            // smoothing window width and height in pixels
//		0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
//
//								   // filter image from grayscale to black and white
//	cv::adaptiveThreshold(matBlurred,                           // input image
//		matThresh,                            // output image
//		255,                                  // make pixels that pass the threshold full white
//		cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
//		cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
//		11,                                   // size of a pixel neighborhood used to calculate threshold value
//		2);                                   // constant subtracted from the mean or weighted mean
//
//	matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image
//
//	std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
//	std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)
//
//	cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
//		ptContours,                             // output contours
//		v4iHierarchy,                           // output hierarchy
//		cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
//		cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
//
//	for (int i = 0; i < ptContours.size(); i++) {               // for each contour
//		ContourWithData contourWithData;                                                    // instantiate a contour with data object
//		contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
//		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
//		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
//		allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
//	}
//
//	for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
//		if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
//			validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
//		}
//	}
//	// sort contours from left to right
//	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);
//
//	std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program
//
//	for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour
//
//																		// draw a green rect around the current char
//		cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
//			validContoursWithData[i].boundingRect,        // rect to draw
//			cv::Scalar(0, 255, 0),                        // green
//			2);                                           // thickness
//
//		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect
//
//		cv::Mat matROIResized;
//		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage
//
//		cv::Mat matROIFloat;
//		matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest
//
//		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
//
//		cv::Mat matCurrentChar(0, 0, CV_32F);
//
//		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!
//
//		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
//
//		strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
//	}
//
//	std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       // show the full string
//
//	cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits
//
//	cv::waitKey(0);                                         // wait for user key press
//
//	return(0);
//}

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv\highgui.h>
//#include <opencv\cv.hpp>
//#include <opencv\cvaux.h>
//#include <iostream>
//#include <vector>
//#include<string.h>
//using namespace std;
//using namespace cv;
//
//int main(int argc, char** argv)
//{
//	cout << "OpenCV Training SVM Automatic Number Plate Recognition\n";
//	cout << "\n";
//
//	char* path_Plates;
//	char* path_NoPlates;
//	int numPlates;
//	int numNoPlates;
//	int imageWidth = 150;
//	int imageHeight = 150;
//
//	//Check if user specify image to process
//	if (1)
//	{
//		numPlates = 12;
//		numNoPlates = 67;
//		path_Plates = "C:/Users/Kietngao2701/Desktop/demo/demo/Positive_Images"; // dương bản	
//		path_NoPlates = "C:/Users/Kietngao2701/Desktop/demo/demo/Negative_Images/"; // âm bản
//
//	}
//	else {
//		cout << "Usage:\n" << argv[0] << " <num Plate Files> <num Non Plate Files> <path to plate folder files> <path to non plate files> \n";
//		return 0;
//	}
//
//	Mat classes;//(numPlates+numNoPlates, 1, CV_32FC1);
//	Mat trainingData;//(numPlates+numNoPlates, imageWidth*imageHeight, CV_32FC1 );
//
//	Mat trainingImages;
//	vector<int> trainingLabels;
//
//	for (int i = 1; i <= numPlates; i++)
//	{
//
//		stringstream ss(stringstream::in | stringstream::out);
//		ss << path_Plates << i << ".jpg";
//		try {
//
//			const char* a = ss.str().c_str();
//			printf("\n%s\n", a);
//			Mat img = imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
//			img = img.clone().reshape(1, 1);
//			imshow("Window",img);//
//			cout<<ss.str();//
//			trainingImages.push_back(img);
//			trainingLabels.push_back(1);
//		}
//		catch (Exception e) { ; }
//	}
//
//	for (int i = 0; i< numNoPlates; i++)
//	{
//		stringstream ss(stringstream::in | stringstream::out);
//		ss << path_NoPlates << i << ".jpg";
//		try
//		{
//			const char* a = ss.str().c_str();
//			printf("\n%s\n", a);
//			Mat img = imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
//			//imshow("Win",img);
//			img = img.clone().reshape(1, 1);
//			trainingImages.push_back(img);
//			trainingLabels.push_back(0);
//			//cout<<ss.str();
//		}
//		catch (Exception e) { ; }
//	}
//
//	Mat(trainingImages).copyTo(trainingData);
//	//trainingData = trainingData.reshape(1,trainingData.rows);
//	trainingData.convertTo(trainingData, CV_32FC1);
//	Mat(trainingLabels).copyTo(classes);
//
//	FileStorage fs("SVM.xml", FileStorage::WRITE);
//	fs << "TrainingData" << trainingData;
//	fs << "classes" << classes;
//	fs.release();
//
//	return 0;
//}