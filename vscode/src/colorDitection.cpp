#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

//画像ファイルの表示
void ditection(char* name)
{
	cv::Mat image = cv::imread(name);	//画像の読み込み
	cv::Mat dst,out;

	if (image.empty())	//画像が正しく読み込めたのかを確認
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}

    cvtColor(image,dst,CV_BGR2HSV);
    cv::inRange(dst,cv::Scalar(40,50,50),cv::Scalar(80,255,255),out);

	cv::imshow("元画像", image);	//画像の表示
    cv::imshow("HSV画像",dst);
    cv::imshow("結果画像",out);

    if (cv::waitKey() == 's')
	{
		cv::imwrite("../image/result/color-result.png",out);
	}

    cv::waitKey();


}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        ditection("../image/angle1.png");
    }
	else{
        ditection(argv[1]);
    }
	
	return 0;
}