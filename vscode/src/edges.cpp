#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

void readImage(char* name)
{
	cv::Mat image = cv::imread(name);
	cv::Mat gray,result;

	if (image.empty())
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}

	cv::imshow("元画像", image);
	cv::cvtColor( image, gray, CV_BGR2GRAY );
    cv::Canny(gray,result,75,25,3);
	cv::imshow("結果画像", result);

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