#include <iostream>
#include <string>
//#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// add int and string
//string operator+(string const &a, int b)
//{
//	ostringstream oss;
//	oss << a << b;
//	return oss.str();
//}

int main()
{
	VideoCapture cap("MVI_1049.avi");
	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}
	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 70;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	int _rows = 310;

	int a = 1;
	

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
		
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		//namedWindow("Original", CV_WINDOW_FREERATIO);
		//imshow("Original", imgOriginal); //show the original image
		resize(imgOriginal, imgOriginal, Size(imgOriginal.rows, _rows));

		Mat imgHSV;
		Mat imgThresholded;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HLS); //Convert the captured frame from BGR to HSV
	//	resize(imgHSV, imgHSV, Size(imgHSV.rows, _rows));

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

		Mat imgGray = cvarrToMat(cannyImg); // Convert IplImage to Mat

											// Reduce the noise so we avoid false circle detection
		GaussianBlur(imgGray, imgGray, Size(9, 9), 2, 2);
		vector<Vec3f> circles;

		// Apply the Hough Transform to find the circles
		HoughCircles(imgGray, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 50, 0, 0);

		// Draw the circles detected

	
		for (size_t i = 0; i < circles.size(); i++)
		{
			string locate = "D:/image";

			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			cout << "center : " << center << "\nradius : " << radius << endl;

			// circle center
			//circle(imgOriginal, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			//circle(imgOriginal, center, radius, Scalar(0, 0, 255), 3, 8, 0);

			Rect crop = Rect(center.x - radius , center.y - radius, 2 * radius, 2 * radius);

			if ((center.x - radius) < 0 || (center.y - radius) < 0) 
				continue;

			Mat cropImg = imgOriginal(crop);
			namedWindow("cropoed", CV_WINDOW_FREERATIO);
			imshow("cropoed", cropImg);
		
			locate += to_string(a) + ".jpg";

			imwrite(locate, cropImg);
			
			cout << cropImg.size() << endl;
			a++;
		//	waitKey(0);
		}

		namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
		imshow("Hough Circle Transform Demo", imgOriginal);

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

