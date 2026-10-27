#pragma once

#include <json/json.h>
#include <opencv2/core/mat.hpp>

#include <string>
#include <string_view>
#include <vector>

struct SamPoint
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

struct SamBox
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
};

struct SamMask
{
    std::vector<SamPoint> points;
    float score = 0;
};

struct SamOutput
{
    std::vector<SamBox> boxes;
    std::vector<SamMask> masks;
};

void load_sam_text();
void load_sam_point();
SamInput make_sam_input(std::string_view bytes, float x, float y, const std::string& tool, const std::string& word);
SamOutput detect_sam_text(const SamInput& input);
SamOutput detect_sam_point(const SamInput& input);
Json::Value sam_output_json(const SamOutput& output);

std::string load_rfdetr();
std::string detect_rfdetr();
