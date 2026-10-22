#include "core/sam3.h"
#include "utils/image.h"

#include <array>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace sam3;

namespace {

constexpr int kContextLength = 32;
constexpr int32_t kSot = 49406;
constexpr int32_t kEot = 49407;
constexpr int32_t kVisualToken = 7195;

const fs::path kModelDir = "backend/data/models/sam3_text";

std::unique_ptr<Ort::Session> image_session;
std::unique_ptr<Ort::Session> language_session;
std::unique_ptr<Ort::Session> decoder_session;

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
    std::array<int32_t, kContextLength> tokens{};
    tokens[0] = kSot;
    tokens[1] = kVisualToken;
    tokens[2] = kEot;
    const int64_t shape[] = {1, kContextLength};
    auto input = Ort::Value::CreateTensor<int32_t>(memory, tokens.data(), tokens.size(), shape, 2);

    const char* input_names[] = {"tokens"};
    const char* output_names[] = {"text_attention_mask", "text_memory", "text_embeds"};
    return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 3);
}

std::vector<Prediction> decode_all(
    Ort::Session& session,
    const Ort::MemoryInfo& memory,
    std::vector<Ort::Value>& image_features,
    std::vector<Ort::Value>& language_features,
    int image_width,
    int image_height)
{
    std::array<float, 4> box_coords{};
    std::array<int32_t, 1> box_labels = {1};
    std::array<bool, 1> box_masks = {true};
    int32_t original_height = image_height;
    int32_t original_width = image_width;
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
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(memory, box_labels.data(), box_labels.size(), label_shape, 2));
    inputs.push_back(Ort::Value::CreateTensor<bool>(memory, box_masks.data(), box_masks.size(), label_shape, 2));
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(memory, &original_height, 1, nullptr, 0));
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(memory, &original_width, 1, nullptr, 0));

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
        "original_height",
        "original_width",
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
    const bool* masks = outputs[2].GetTensorData<bool>();
    const auto mask_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    const int mask_height = static_cast<int>(mask_shape[2]);
    const int mask_width = static_cast<int>(mask_shape[3]);

    std::vector<Prediction> predictions;
    predictions.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        Prediction prediction;
        prediction.x1 = boxes[i * 4 + 0];
        prediction.y1 = boxes[i * 4 + 1];
        prediction.x2 = boxes[i * 4 + 2];
        prediction.y2 = boxes[i * 4 + 3];
        prediction.score = scores[i];

        const size_t offset = static_cast<size_t>(i) * mask_height * mask_width;
        cv::Mat mask(mask_height, mask_width, CV_8UC1, const_cast<bool*>(masks + offset));
        prediction.mask = mask.clone();
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

}  // namespace

void load_sam_text()
{
    image_session = open_session(kModelDir / "sam3_image_encoder.onnx", gpu_options());
    language_session = open_session(kModelDir / "sam3_language_encoder.onnx", cpu_options());
    decoder_session = open_session(kModelDir / "sam3_decoder.onnx", cpu_options());
}

SamOutput detect_sam_text(const SamInput& input)
{
    const std::string tool = normalized_tool(input.tool);
    if (!image_session || !language_session || !decoder_session) {
        throw std::runtime_error("SAM3 Text 모델을 먼저 불러와야 합니다.");
    }
    const cv::Mat rgb = input_rgb(input);
    const ImageSize size = image_size(input.image);

    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto image_features = encode_image(*image_session, memory, rgb);
    auto language_features = encode_language(*language_session, memory);
    const auto predictions = decode_all(*decoder_session, memory, image_features, language_features, size.width, size.height);
    return to_output(predictions, tool);
}
