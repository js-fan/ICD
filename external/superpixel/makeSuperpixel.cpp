#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <omp.h>
#include "segment.h"

typedef unsigned int uint;

void make_superpixel(const char* src, const char* dst, float k) {
    cv::Mat image = cv::imread(src);
    cv::Mat segment_32SC1;
    segment(image, segment_32SC1, 0.8, k, k);
    
    cv::Mat segment_8UC3(segment_32SC1.rows, segment_32SC1.cols, CV_8UC3);
    for (int i = 0; i < segment_32SC1.rows; ++i) {
        for (int j = 0; j < segment_32SC1.cols; ++j) {
            int num = segment_32SC1.at<int>(i, j);
            cv::Vec3b &target = segment_8UC3.at<cv::Vec3b>(i, j);
            target.val[0] = (uint)num;
            num >>= 8;
            target.val[1] = (uint)num;
            num >>= 8;
            target.val[2] = (uint)num;
        }
    }
    cv::imwrite(dst, segment_8UC3);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./makeSuperpixel [data_list.txt] [image_dir] [target_dir] [k]\n";
        return -1;
    }

    // load names 
    std::ifstream data_list_file(argv[1]);
    if (!data_list_file.is_open()) {
        std::cerr << "Cannot find file " << argv[1] << std::endl;
        return -1;
    }

    std::string source_dir(argv[2]);
    std::string target_dir(argv[3]);
    if (source_dir.back() != '/')
        source_dir += '/';
    if (target_dir.back() != '/')
        target_dir += '/';
    boost::filesystem::create_directories(target_dir);
    std::cout << "Created dir: " << target_dir << std::endl;

    std::string name;
    std::vector<std::pair<std::string, std::string>> src_dst_names;
    while (std::getline(data_list_file, name)) {
        src_dst_names.emplace_back(
                source_dir + name + ".jpg",
                target_dir + name + ".png"
        );
    }

    // begin
    float k = std::stof(argv[4]);

    #ifdef use_openmp_
    #pragma omp parallel for
    #endif
    for (int i = 0; i < src_dst_names.size(); ++i) {
        make_superpixel(src_dst_names[i].first.c_str(), src_dst_names[i].second.c_str(), k);
    }

    return 0;
}
