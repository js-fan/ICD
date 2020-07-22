#ifndef SEGMENT_H_
#define SEGMENT_H_


#include <opencv2/opencv.hpp>


/*
 * OpenCV wrapper
 *
 * input:   CV_8UC3  (uchar)
 * output:  CV_32SC1 (int)
 * return:  int, num of components
 */
int segment(const cv::Mat &input, cv::Mat &output, float sigma, float c, int min_size);


/*
 * Draw random color
 *
 * input:   CV_32SC1 (int)
 * output:  CV_8UC3  (uchar)
 */
void draw_segment(const cv::Mat &input_comp, cv::Mat &output);


#endif
