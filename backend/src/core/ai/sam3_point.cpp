#include "core/sam3.h"
#include "utils/image.h"

#include <opencv2/opencv.hpp>

#include <array>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace sam3;

namespace {

const fs::path kModelDir = "backend/data/models/sam3_point";

std::unique_ptr<Ort::Session> image_session;
std::unique_ptr<Ort::Session> prompt_session;

std::vector<Ort::Value> encode_image(Ort::Session& session, const Ort::MemoryInfo& memory, const cv::Mat& rgb)
{
    const std::vector<uint8_t> chw = rgb_to_chw1008(rgb);
    std::vector<float> pixels(chw.size());
    for (size_t i = 0; i < chw.size(); ++i) {
        pixels[i] = static_cast<float>(chw[i]) / 127.5f - 1.f;
    }
    const int64_t shape[] = {1, 3, kImageSize, kImageSize};
    auto input = Ort::Value::CreateTensor<float>(memory, pixels.data(), pixels.size(), shape, 4);

    const char* input_names[] = {"pixel_values"};
    const char* output_names[] = {"image_embeddings.0", "image_embeddings.1", "image_embeddings.2"};
    return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 3);
}

std::vector<Prediction> decode_point(
    Ort::Session& session,
    const Ort::MemoryInfo& memory,
    std::vector<Ort::Value>& features,
    float click_x,
    float click_y,
    int image_width,
    int image_height)
{
    std::array<float, 2> point = {click_x * kImageSize, click_y * kImageSize};
    std::array<int64_t, 1> label = {1};
    std::array<float, 4> no_box{};
    const int64_t point_shape[] = {1, 1, 1, 2};
    const int64_t label_shape[] = {1, 1, 1};
    const int64_t box_shape[] = {1, 0, 4};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, point.data(), point.size(), point_shape, 4));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, label.data(), label.size(), label_shape, 3));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, no_box.data(), 0, box_shape, 3));
    inputs.push_back(view_float(memory, features[0]));
    inputs.push_back(view_float(memory, features[1]));
    inputs.push_back(view_float(memory, features[2]));

    const char* input_names[] = {
        "input_points",
        "input_labels",
        "input_boxes",
        "image_embeddings.0",
        "image_embeddings.1",
        "image_embeddings.2",
    };
    const char* output_names[] = {"iou_scores", "pred_masks"};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        2);

    const float* scores = outputs[0].GetTensorData<float>();
    const float* masks = outputs[1].GetTensorData<float>();
    const auto mask_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const int64_t count = mask_shape[2];
    const int mask_height = static_cast<int>(mask_shape[3]);
    const int mask_width = static_cast<int>(mask_shape[4]);

    int64_t best = 0;
    for (int64_t i = 1; i < count; ++i) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }

    const size_t offset = static_cast<size_t>(best) * mask_height * mask_width;
    cv::Mat low_res(mask_height, mask_width, CV_32FC1, const_cast<float*>(masks + offset));
    cv::Mat resized;
    cv::resize(low_res, resized, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

    Prediction prediction;
    cv::compare(resized, 0.f, prediction.mask, cv::CMP_GT);
    if (cv::countNonZero(prediction.mask) == 0) {
        return {};
    }
    const cv::Rect box = cv::boundingRect(prediction.mask);
    prediction.x1 = static_cast<float>(box.x);
    prediction.y1 = static_cast<float>(box.y);
    prediction.x2 = static_cast<float>(box.x + box.width);
    prediction.y2 = static_cast<float>(box.y + box.height);
    prediction.score = scores[best];
    return {std::move(prediction)};
}

}  // namespace

void load_sam_point()
{
    image_session = open_session(kModelDir / "vision_encoder_q4f16.onnx", gpu_options());
    prompt_session = open_session(kModelDir / "prompt_encoder_mask_decoder_int8.onnx", cpu_options());
}

SamOutput detect_sam_point(const SamInput& input)
{
    const std::string tool = normalized_tool(input.tool);
    if (!image_session || !prompt_session) {
        throw std::runtime_error("SAM3 Point 모델을 먼저 불러와야 합니다.");
    }
    const cv::Mat rgb = input_rgb(input);
    const ImageSize size = image_size(input.image);

    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto features = encode_image(*image_session, memory, rgb);
    const auto predictions = decode_point(*prompt_session, memory, features, input.x, input.y, size.width, size.height);
    return to_output(predictions, tool);
}
