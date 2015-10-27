// retinextest.cpp : Defines the entry point for the console application.
//

#include "lum_retinex.h"
#include <opencv2/opencv.hpp>

const float threshold = 0.13;

int main(int argc, char* argv[])
{
	cv::Mat input = cv::imread("img/input.ppm", CV_LOAD_IMAGE_ANYDEPTH |CV_LOAD_IMAGE_COLOR);
	input.convertTo(input, CV_32FC3, 1.0 / USHRT_MAX);

	int w = input.cols;
	int h = input.rows;
	cv::Mat reflectance(h, w, CV_32FC3);
	cv::Mat shading(h, w, CV_32FC1);
	lum::retinex_decomp rdecomp(w, h);
	rdecomp.solve_rgb(threshold, (const float*)input.data, (float*)reflectance.data, (float*)shading.data);

	cv::imshow("input", input);
	cv::imshow("shading", shading);
	cv::imshow("reflectance", reflectance);
	cv::waitKey(0);
	return 0;
}

