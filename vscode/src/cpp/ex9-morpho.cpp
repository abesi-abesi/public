#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

//画像ファイルの表示
void readImage(char* fileName,char* outputName)
{
	cv::Mat image = cv::imread(fileName);	//画像の読み込み
    cv::Mat out,gray;

	if (image.empty())	//画像が正しく読み込めたのかを確認
	{
		cout << "画像ファイルを読み込めませんでした．" << endl;
		return;
	}
	
    cv::imshow("original", image);
    cv::cvtColor(image,gray,CV_BGR2GRAY);
    double outThresh = threshold(gray,out,128,255,cv::THRESH_OTSU);
    cv::imshow("output",out);

    while (cv::waitKey() != 'q'){

    switch(cv::waitKey()){
        case 'e':
        cv::erode(out,out,cv::Mat());
        cout << "erode" << endl;
        break;

        case 'd':
        cv:dilate(out,out,cv::Mat());
        cout << "dilate" << endl;
        break;

        case'r':
        cv::destroyWindow("output");
        image = cv::imread(fileName);
        cv::cvtColor(image,gray,CV_BGR2GRAY);
        outThresh = threshold(gray,out,128,255,cv::THRESH_OTSU);
        cout << "reload" << endl;
        break;

        case 's':
        cv::imwrite(outputName,out);
        cout << "save" << endl;
        break;   
        }
        cv::imshow("output",out);
    }
}

int main(int argc, char** argv)
{
    char* args[3] = {argv[0],"string.jpg","cstring.jpg"};
    if (argc >= 2) args[1] = argv[1];
    if (argc >= 3) args[2] = argv[2];
	readImage(args[1],args[2]);
	return 0;
}
