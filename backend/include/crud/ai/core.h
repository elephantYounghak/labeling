#pragma once

#include <json/json.h>
#include <opencv2/core/mat.hpp>

#include <map>
#include <string>
#include <string_view>
#include <vector>

struct Point
{
    float x = 0;
    float y = 0;
};

struct SamInput
{
    cv::Mat image;
    float x = 0;
    float y = 0;
    std::string tool;
    std::string word;
};

struct RfdetrInput
{
    cv::Mat image;

};

struct Box
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
};

struct Mask
{
    std::vector<Point> points;
    float score = 0;
};

struct Output
{
    std::vector<Box> boxes;
    std::vector<Mask> masks;
};

using Outputs = std::map<std::string, Output>;

void load_sam_text();
void load_sam_point();
SamInput make_sam_input(std::string_view bytes, float x, float y, const std::string& tool, const std::string& word);
Output detect_sam_text(const SamInput& input);
Output detect_sam_point(const SamInput& input);
Json::Value sam_output_json(const Output& output);
