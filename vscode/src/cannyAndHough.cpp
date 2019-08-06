#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

void cannyAndHough(cv::Mat src, cv::Mat dst, double rho, 
		double theta, int threshold, double minL, double minGap)
	{
	using namespace cv;

	Mat gsrc,edges;
	cvtColor( src, gsrc, COLOR_BGR2GRAY );
    Canny( gsrc, edges, 75, 25, 3 );
	imshow("エッジ画像",edges);
    cvtColor( edges, dst, COLOR_GRAY2BGR );
    vector<Vec4i> lines;
    HoughLinesP( edges, lines, rho, theta, threshold, minL, minGap);
    for( size_t i = 0; i < lines.size(); i++ ) {
        line( dst, Point(lines[i][0], lines[i][1]),
        Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 1, 8 );
    }
	if (cv::waitKey() == 'e')
	{
		cv::imwrite("../image/result/new-edges-result.png",edges);
	}
}

void readImage(char* name)
{
	cv::Mat image = cv::imread(name);
    int threshold = 200;
	double minL = 100;
	cv::Mat result = image.clone();

	if (image.empty())
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}

	cv::imshow("元画像", image);
    cannyAndHough(image, result, 1, CV_PI/180, threshold, minL, 10);
	cv::imshow("結果画像", result);
	
	if (cv::waitKey() == 's')
	{
		cv::imwrite("../image/result/new-hough-result.png",result);
	}

	cv::waitKey();
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        readImage("../image/sample.jpg");
    }
	else{
        readImage(argv[1]);
    }
	
	return 0;
}