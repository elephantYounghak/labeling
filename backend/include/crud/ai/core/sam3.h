#pragma once

#include <json/json.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
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

using SamOutputs = std::map<std::string, SamOutput>;

namespace sam3 {

constexpr int kImageSize = 1008;

struct Prediction
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
    cv::Mat mask;
};

Ort::SessionOptions cpu_options();
Ort::SessionOptions gpu_options();
std::unique_ptr<Ort::Session> open_session(const std::filesystem::path& path, const Ort::SessionOptions& options);
std::string normalized_tool(const std::string& tool);
cv::Mat input_rgb(const SamInput& input);
std::vector<uint8_t> rgb_to_chw1008(const cv::Mat& rgb);
Ort::Value view_float(const Ort::MemoryInfo& memory, Ort::Value& source);
Ort::Value view_bool(const Ort::MemoryInfo& memory, Ort::Value& source);
SamOutput to_output(const std::vector<Prediction>& predictions, const std::string& tool);
SamInput make_input(std::string_view bytes, float x, float y, const std::string& tool, const std::string& word);
Json::Value output_json(const SamOutput& output);

}  // namespace sam3
