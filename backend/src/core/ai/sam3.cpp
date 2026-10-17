#include "core/ai.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kImageSize = 1008;
constexpr int kMaskSize = 288;
constexpr int kContextLength = 32;
constexpr float kClickBoxPx = 32.f;
constexpr int64_t kSot = 49406;
constexpr int64_t kEot = 49407;
constexpr int64_t kVisualToken = 7195;

const fs::path kModelDir = "backend/data/models/sam3";

struct Prediction
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
    cv::Mat mask;
};

Ort::Env& ort_env()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam3");
    return env;
}

std::unique_ptr<Ort::Session> image_session;
std::unique_ptr<Ort::Session> language_session;
std::unique_ptr<Ort::Session> decoder_session;

void require_file(const fs::path& path)
{
    if (!fs::exists(path)) {
        throw std::runtime_error("파일이 없습니다: " + path.string());
    }
}

std::unique_ptr<Ort::Session> open_session(const fs::path& path)
{
    require_file(path);
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef _WIN32
    return std::make_unique<Ort::Session>(ort_env(), path.wstring().c_str(), options);
#else
    return std::make_unique<Ort::Session>(ort_env(), path.string().c_str(), options);
#endif
}

std::string normalized_tool(const std::string& tool)
{
    if (tool == "bbox" || tool == "polygon" || tool == "brush") {
        return tool;
    }
    throw std::runtime_error("도구는 bbox, polygon, brush 중 하나여야 합니다.");
}

std::vector<uint8_t> rgb_to_chw1008(const cv::Mat& rgb)
{
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_LINEAR);

    std::vector<uint8_t> chw(3 * kImageSize * kImageSize);
    for (int y = 0; y < kImageSize; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kImageSize; ++x) {
            chw[0 * kImageSize * kImageSize + y * kImageSize + x] = row[x][0];
            chw[1 * kImageSize * kImageSize + y * kImageSize + x] = row[x][1];
            chw[2 * kImageSize * kImageSize + y * kImageSize + x] = row[x][2];
        }
    }
    return chw;
}

Ort::Value view_float(const Ort::MemoryInfo& memory, Ort::Value& source)
{
    const auto info = source.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    return Ort::Value::CreateTensor<float>(
        memory,
        source.GetTensorMutableData<float>(),
        info.GetElementCount(),
        shape.data(),
        shape.size());
}

Ort::Value view_bool(const Ort::MemoryInfo& memory, Ort::Value& source)
{
    const auto info = source.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    return Ort::Value::CreateTensor<bool>(
        memory,
        source.GetTensorMutableData<bool>(),
        info.GetElementCount(),
        shape.data(),
        shape.size());
}

std::vector<Ort::Value> encode_image(Ort::Session& session, const Ort::MemoryInfo& memory, const cv::Mat& rgb)
{
    std::vector<uint8_t> chw = rgb_to_chw1008(rgb);
    const int64_t shape[] = {3, kImageSize, kImageSize};
    auto input = Ort::Value::CreateTensor<uint8_t>(memory, chw.data(), chw.size(), shape, 3);

    const char* input_names[] = {"image"};
    const char* output_names[] = {
        "vision_pos_enc_0",
        "vision_pos_enc_1",
        "vision_pos_enc_2",
        "backbone_fpn_0",
        "backbone_fpn_1",
        "backbone_fpn_2",
    };
    return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 6);
}

std::vector<Ort::Value> encode_language(Ort::Session& session, const Ort::MemoryInfo& memory)
{
    std::array<int64_t, kContextLength> tokens{};
    tokens[0] = kSot;
    tokens[1] = kVisualToken;
    tokens[2] = kEot;
    const int64_t shape[] = {1, kContextLength};
    auto input = Ort::Value::CreateTensor<int64_t>(memory, tokens.data(), tokens.size(), shape, 2);

    const char* input_names[] = {"tokens"};
    const char* output_names[] = {"text_attention_mask", "text_memory", "text_embeds"};
    return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 3);
}

std::vector<Prediction> decode_click(
    Ort::Session& session,
    const Ort::MemoryInfo& memory,
    std::vector<Ort::Value>& image_features,
    std::vector<Ort::Value>& language_features,
    float click_x,
    float click_y,
    int image_width,
    int image_height)
{
    const float cx = click_x / static_cast<float>(image_width);
    const float cy = click_y / static_cast<float>(image_height);
    const float w = kClickBoxPx / static_cast<float>(image_width);
    const float h = kClickBoxPx / static_cast<float>(image_height);

    std::array<float, 4> box_coords = {cx, cy, w, h};
    std::array<int64_t, 1> box_labels = {1};
    std::array<bool, 1> box_masks = {false};
    const int64_t coord_shape[] = {1, 1, 4};
    const int64_t label_shape[] = {1, 1};

    std::vector<Ort::Value> inputs;
    inputs.push_back(view_float(memory, image_features[3]));
    inputs.push_back(view_float(memory, image_features[4]));
    inputs.push_back(view_float(memory, image_features[5]));
    inputs.push_back(view_float(memory, image_features[2]));
    inputs.push_back(view_bool(memory, language_features[0]));
    inputs.push_back(view_float(memory, language_features[1]));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, box_coords.data(), box_coords.size(), coord_shape, 3));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, box_labels.data(), box_labels.size(), label_shape, 2));
    inputs.push_back(Ort::Value::CreateTensor<bool>(memory, box_masks.data(), box_masks.size(), label_shape, 2));

    const char* input_names[] = {
        "backbone_fpn_0",
        "backbone_fpn_1",
        "backbone_fpn_2",
        "vision_pos_enc_2",
        "language_mask",
        "language_features",
        "box_coords",
        "box_labels",
        "box_masks",
    };
    const char* output_names[] = {"boxes", "scores", "masks"};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        3);

    const auto box_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int64_t count = box_shape.empty() ? 0 : box_shape[0];
    const float* boxes = outputs[0].GetTensorData<float>();
    const float* scores = outputs[1].GetTensorData<float>();
    const float* masks = outputs[2].GetTensorData<float>();

    std::vector<Prediction> predictions;
    predictions.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        Prediction prediction;
        prediction.x1 = boxes[i * 4 + 0] * static_cast<float>(image_width);
        prediction.y1 = boxes[i * 4 + 1] * static_cast<float>(image_height);
        prediction.x2 = boxes[i * 4 + 2] * static_cast<float>(image_width);
        prediction.y2 = boxes[i * 4 + 3] * static_cast<float>(image_height);
        prediction.score = scores[i];

        cv::Mat low_res(kMaskSize, kMaskSize, CV_32FC1, const_cast<float*>(masks + i * kMaskSize * kMaskSize));
        cv::Mat resized;
        cv::resize(low_res, resized, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
        cv::compare(resized, 0.5, prediction.mask, cv::CMP_GT);
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

std::vector<SamPoint> contour_points(const cv::Mat& mask, bool simplify)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }

    size_t best = 0;
    double best_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }

    std::vector<cv::Point> shape = contours[best];
    if (simplify) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(shape, approx, 0.01 * cv::arcLength(shape, true), true);
        if (approx.size() >= 3) {
            shape = std::move(approx);
        }
    }

    std::vector<SamPoint> points;
    points.reserve(shape.size());
    for (const cv::Point& point : shape) {
        points.push_back({static_cast<float>(point.x), static_cast<float>(point.y)});
    }
    return points;
}

void append_detection(SamOutput& output, const Prediction& prediction, const std::string& tool)
{
    if (tool == "bbox") {
        output.boxes.push_back({prediction.x1, prediction.y1, prediction.x2, prediction.y2, prediction.score});
        return;
    }
    SamMask mask;
    mask.score = prediction.score;
    mask.points = contour_points(prediction.mask, tool == "polygon");
    if (!mask.points.empty()) {
        output.masks.push_back(std::move(mask));
    }
}

}  // namespace

void load_sam()
{
    image_session = open_session(kModelDir / "sam3_image_encoder.onnx");
    language_session = open_session(kModelDir / "sam3_language_encoder.onnx");
    decoder_session = open_session(kModelDir / "sam3_decoder.onnx");
}

SamOutput detect_sam(const SamInput& input)
{
    const std::string shape = normalized_tool(input.tool);
    if (!image_session || !language_session || !decoder_session) {
        throw std::runtime_error("SAM 모델을 먼저 불러와야 합니다.");
    }

    cv::Mat bgr = cv::imread(input.image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("이미지를 열지 못했습니다: " + input.image_path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto image_features = encode_image(*image_session, memory, rgb);
    auto language_features = encode_language(*language_session, memory);
    const auto predictions = decode_click(*decoder_session, memory, image_features, language_features, input.x, input.y, bgr.cols, bgr.rows);

    SamOutput detections;
    for (const Prediction& prediction : predictions) {
        append_detection(detections, prediction, shape);
    }
    return detections;
}
