#pragma once

#include <opencv2/core/mat.hpp>

#include <string>
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

void load_sam();
SamOutput detect_sam(const SamInput& input);

std::string load_rfdetr();
std::string detect_rfdetr();
