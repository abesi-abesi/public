#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

//画像ファイルの表示
void morpho(char* name)
{
	cv::Mat image = cv::imread(name);	//画像の読み込み
	cv::Mat out = image;

	if (image.empty())	//画像が正しく読み込めたのかを確認
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}

	cv::imshow("元画像", image);	//画像の表示
	cv::erode(out,out,cv::Mat(),cv::Point(-1,-1),15);
	cv::dilate(out,out,cv::Mat(),cv::Point(-1,-1),120);
	cv::erode(out,out,cv::Mat(),cv::Point(-1,-1),105);
	cv::imshow("結果画像",out);

	while (cv::waitKey() != 'q')
	{
		switch(cv::waitKey()){
			// case 'e':
			// cv::erode(out,out,cv::Mat(),cv::Point(-1,-1),15);
			// cout << "erode" << endl;
			// break;

			// case 'd':
			// cv::dilate(out,out,cv::Mat(),cv::Point(-1,-1),15);
			// cout << "dilate" << endl;
			// break;

			case 's':
			cv::imwrite("../image/result/mask-result.png",out);

		}
		// cv::imshow("結果画像",out);
	}
	
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        morpho("../image/angle1.png");
    }
	else{
        morpho(argv[1]);
    }
	
	return 0;
}
