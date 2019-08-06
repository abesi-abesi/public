#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

//画像ファイルの表示
void ditection(char* fileName,char* maskName)
{
	cv::Mat image = cv::imread(fileName);	//画像の読み込み
	cv::Mat mask = cv::imread(maskName);
    cv::Mat dst;

	if (image.empty())	//画像が正しく読み込めたのかを確認
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}

    image.copyTo(dst,mask);

    cv::imshow("元画像",image);
    cv::imshow("マスク",mask);
    cv::imshow("結果画像",dst);

    if (cv::waitKey() == 's')
	{
		cv::imwrite("../image/result/maskedImage.png",dst);
	}

    cv::waitKey();


}

int main(int argc, char** argv)
{
    char*args[3] = {argv[0],"../image/angle1.png","../image/result/mask-result1.png"};
    if (argc >= 2) args[1] = argv[1];
    if (argc >= 3) args[2] = argv[2];
    ditection(args[1],args[2]);
	return 0;
}