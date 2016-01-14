#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <CL/cl.hpp>


#define BUSY 0
#define EMPTY 255

struct result
{
	int xpos, ypos;
	float SAD;
};

class TemplateMatch {
public:
	int HEIGHT,WIDTH;


	TemplateMatch(  cv::Mat img );
	cv::Mat image;

	result check(cv::Mat _template, int t_rows, int t_cols);
};


void loadDataMatToUchar(uchar *data,cv::Mat &image,int nchannels)
{
    int width = image.cols;
    int height = image.rows;
    for (int y=0; y<height;y++)
	{
        for (int x = 0 ; x<width ; x++)
		{
            data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 0] = image.at<uchar>(y,x);
            if (nchannels==3)
			{
                data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 1] = image.at<uchar>(y,x);
                data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 2] = image.at<uchar>(y,x);
            }
        }
    }
}

void ucharToMat(uchar *data,cv::Mat &image)
{
	for (int y=0; y<image.rows;y++)
	{
		for (int x = 0 ; x<image.cols ; x++)
		{
			image.at<uchar>(y,x) = data[(long)y * (long)image.cols + x] ;
		}
	}
}


TemplateMatch::TemplateMatch( cv::Mat img )
{
	WIDTH = img.cols;
	HEIGHT = img.rows;
	image = img;


}

result TemplateMatch::check(cv::Mat _template, int t_rows, int t_cols)
{

	uchar* imageData = new uchar[image.cols * image.rows];
	uchar* templateData = new uchar[_template.cols * _template.rows];

    result res;
	res.SAD = 100000000;
	WIDTH = image.cols;
	HEIGHT = image.rows;


	loadDataMatToUchar(imageData,image,1);
	loadDataMatToUchar(templateData,_template,1);


// loop through the search image
    for ( int y = 0; y <= HEIGHT - t_rows; y++ )
	{

	for ( int x =0; x <= WIDTH - t_cols; x++ )
   {
       float SAD = 0.0;

	// loop through the template image

		for ( int y1 = 0; y1 < t_rows; y1++ )
            for ( int x1 = 0; x1 < t_cols; x1++ )
			{

				int p_SearchIMG = imageData[(y+y1) * WIDTH + (x+x1)];
                int p_TemplateIMG = templateData[y1 *  t_cols + x1];

                SAD += abs( p_SearchIMG - p_TemplateIMG );
            }

        // save the best found position
        if ( res.SAD > SAD )
		{
            res.SAD = SAD;
            // give me min SAD
			res.xpos = x;
            res.ypos = y;
			//std::cout<< " x "<< x <<" y "<<y << " SAD "<< SAD << "\n";

        }
    }
}

	return res;
}
