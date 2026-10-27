#include "crud/ai/core/sam3_text.h"
#include "crud/ai/core/sam3.h"
#include "utils/image.h"

#include <json/json.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace sam3_text {
using namespace sam3;

constexpr int kContextLength = 32;
constexpr int64_t kSot = 49406;
constexpr int64_t kEot = 49407;

const fs::path kModelDir = "backend/data/models/sam3_text";

namespace {

struct Tokens
{
    std::array<int64_t, kContextLength> ids{};
    std::array<int64_t, kContextLength> attention{};
};

struct LanguageFeatures
{
    std::vector<Ort::Value> values;
    std::array<int64_t, kContextLength> attention{};
};

struct ClipVocab
{
    std::unordered_map<std::string, int64_t> ids;
    std::unordered_map<std::string, int> ranks;
    std::array<std::string, 256> bytes{};
};

std::string utf8_codepoint(uint32_t codepoint)
{
    std::string text;
    if (codepoint < 0x80) {
        text.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        text.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        text.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return text;
}

std::array<std::string, 256> byte_encoder()
{
    std::vector<int> bytes;
    for (int value = '!'; value <= '~'; ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xA1; value <= 0xAC; ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xAE; value <= 0xFF; ++value) {
        bytes.push_back(value);
    }
    std::vector<int> codepoints = bytes;
    std::array<char, 256> seen{};
    for (int value : bytes) {
        seen[static_cast<size_t>(value)] = 1;
    }
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (seen[static_cast<size_t>(value)]) {
            continue;
        }
        bytes.push_back(value);
        codepoints.push_back(256 + extra);
        ++extra;
    }

    std::array<std::string, 256> encoder;
    for (size_t i = 0; i < bytes.size(); ++i) {
        encoder[static_cast<size_t>(bytes[i])] = utf8_codepoint(static_cast<uint32_t>(codepoints[i]));
    }
    return encoder;
}

std::string cleaned_prompt(const std::string& word)
{
    std::string text;
    bool pending_space = false;
    for (unsigned char value : word) {
        if (value < 128) {
            value = static_cast<unsigned char>(std::tolower(value));
        }
        if (value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' || value == '\v') {
            pending_space = !text.empty();
            continue;
        }
        if (pending_space) {
            text.push_back(' ');
            pending_space = false;
        }
        text.push_back(static_cast<char>(value));
    }
    return text;
}

size_t codepoint_size(unsigned char lead)
{
    if (lead < 0x80) {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

uint32_t codepoint_at(const std::string& text, size_t index)
{
    const auto lead = static_cast<unsigned char>(text[index]);
    const size_t size = codepoint_size(lead);
    if (index + size > text.size() || size == 1) {
        return lead;
    }
    if (size == 2) {
        return (static_cast<uint32_t>(lead & 0x1F) << 6) | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
    }
    if (size == 3) {
        return (static_cast<uint32_t>(lead & 0x0F) << 12)
            | (static_cast<uint32_t>(static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6)
            | (static_cast<unsigned char>(text[index + 2]) & 0x3F);
    }
    return (static_cast<uint32_t>(lead & 0x07) << 18)
        | (static_cast<uint32_t>(static_cast<unsigned char>(text[index + 1]) & 0x3F) << 12)
        | (static_cast<uint32_t>(static_cast<unsigned char>(text[index + 2]) & 0x3F) << 6)
        | (static_cast<unsigned char>(text[index + 3]) & 0x3F);
}

bool is_letter(uint32_t codepoint)
{
    return (codepoint >= 'a' && codepoint <= 'z') || codepoint > 127;
}

bool is_digit(uint32_t codepoint)
{
    return codepoint >= '0' && codepoint <= '9';
}

std::vector<std::string> split_prompt(const std::string& text)
{
    static const char* kContractions[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
    std::vector<std::string> parts;
    for (size_t i = 0; i < text.size();) {
        if (text[i] == ' ') {
            ++i;
            continue;
        }
        bool contraction = false;
        for (const char* item : kContractions) {
            const size_t length = std::char_traits<char>::length(item);
            if (text.compare(i, length, item) == 0) {
                parts.emplace_back(item, length);
                i += length;
                contraction = true;
                break;
            }
        }
        if (contraction) {
            continue;
        }

        const uint32_t codepoint = codepoint_at(text, i);
        const size_t start = i;
        i += codepoint_size(static_cast<unsigned char>(text[start]));
        if (is_letter(codepoint)) {
            while (i < text.size() && is_letter(codepoint_at(text, i))) {
                i += codepoint_size(static_cast<unsigned char>(text[i]));
            }
        } else if (!is_digit(codepoint)) {
            while (i < text.size()) {
                const uint32_t next = codepoint_at(text, i);
                if (next == ' ' || is_letter(next) || is_digit(next)) {
                    break;
                }
                i += codepoint_size(static_cast<unsigned char>(text[i]));
            }
        }
        parts.emplace_back(text, start, i - start);
    }
    return parts;
}

std::vector<std::string> merge_token(const std::string& token, const std::unordered_map<std::string, int>& ranks)
{
    std::vector<std::string> word;
    word.reserve(token.size());
    for (size_t i = 0; i < token.size();) {
        const size_t size = codepoint_size(static_cast<unsigned char>(token[i]));
        const size_t end = std::min(token.size(), i + size);
        word.emplace_back(token, i, end - i);
        i = end;
    }
    if (word.empty()) {
        return word;
    }
    word.back() += "</w>";

    while (word.size() > 1) {
        int best_rank = -1;
        std::string best_first;
        std::string best_second;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            const auto found = ranks.find(word[i] + "\n" + word[i + 1]);
            if (found == ranks.end()) {
                continue;
            }
            if (best_rank < 0 || found->second < best_rank) {
                best_rank = found->second;
                best_first = word[i];
                best_second = word[i + 1];
            }
        }
        if (best_rank < 0) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == best_first && word[i + 1] == best_second) {
                merged.push_back(best_first + best_second);
                i += 2;
            } else {
                merged.push_back(word[i]);
                ++i;
            }
        }
        word = std::move(merged);
    }
    return word;
}

ClipVocab load_vocab()
{
    ClipVocab vocab;
    vocab.bytes = byte_encoder();
    vocab.ids.reserve(50000);
    vocab.ranks.reserve(50000);

    std::ifstream vocab_file(kModelDir / "vocab.json");
    Json::Value root;
    std::string errors;
    if (!vocab_file || !Json::parseFromStream(Json::CharReaderBuilder(), vocab_file, &root, &errors) || !root.isObject()) {
        throw std::runtime_error("vocab.json을 읽지 못했습니다.");
    }
    for (const std::string& name : root.getMemberNames()) {
        vocab.ids.emplace(name, root[name].asInt64());
    }

    std::ifstream merge_file(kModelDir / "merges.txt");
    if (!merge_file) {
        throw std::runtime_error("merges.txt를 읽지 못했습니다.");
    }
    std::string line;
    int rank = 0;
    while (std::getline(merge_file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t split = line.find(' ');
        if (split == std::string::npos || split + 1 >= line.size()) {
            continue;
        }
        vocab.ranks.emplace(line.substr(0, split) + "\n" + line.substr(split + 1), rank);
        ++rank;
    }
    if (vocab.ids.empty() || vocab.ranks.empty()) {
        throw std::runtime_error("CLIP 토크나이저를 만들지 못했습니다.");
    }
    return vocab;
}

const ClipVocab& tokenizer()
{
    static const ClipVocab vocab = load_vocab();
    return vocab;
}

Tokens tokenize(const std::string& word)
{
    const std::string text = cleaned_prompt(word);
    if (text.empty()) {
        throw std::runtime_error("검출할 라벨 이름이 없습니다.");
    }

    const ClipVocab& vocab = tokenizer();
    std::vector<int64_t> content;
    content.reserve(kContextLength);
    for (const std::string& part : split_prompt(text)) {
        std::string mapped;
        for (unsigned char value : part) {
            mapped += vocab.bytes[value];
        }
        for (const std::string& piece : merge_token(mapped, vocab.ranks)) {
            const auto found = vocab.ids.find(piece);
            if (found == vocab.ids.end()) {
                throw std::runtime_error("라벨을 CLIP 토큰으로 바꾸지 못했습니다.");
            }
            if (content.size() >= static_cast<size_t>(kContextLength - 2)) {
                break;
            }
            content.push_back(found->second);
        }
    }
    if (content.empty()) {
        throw std::runtime_error("검출할 라벨 이름이 없습니다.");
    }

    Tokens tokens;
    tokens.ids[0] = kSot;
    tokens.attention[0] = 1;
    for (size_t i = 0; i < content.size(); ++i) {
        tokens.ids[i + 1] = content[i];
        tokens.attention[i + 1] = 1;
    }
    const size_t end = content.size() + 1;
    tokens.ids[end] = kEot;
    tokens.attention[end] = 1;
    for (size_t i = end + 1; i < static_cast<size_t>(kContextLength); ++i) {
        tokens.ids[i] = kEot;
    }
    return tokens;
}

float positive_score(float logit)
{
    if (logit >= 0.f) {
        const float exponent = std::exp(-logit);
        return 1.f / (1.f + exponent);
    }
    const float exponent = std::exp(logit);
    return exponent / (1.f + exponent);
}

}  // namespace

std::unique_ptr<Ort::Session> image_session;
std::unique_ptr<Ort::Session> language_session;
std::unique_ptr<Ort::Session> decoder_session;

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
    const char* output_names[] = {
        "fpn_hidden_state_0",
        "fpn_hidden_state_1",
        "fpn_hidden_state_2",
        "fpn_hidden_state_3",
        "fpn_position_encoding_0",
        "fpn_position_encoding_1",
        "fpn_position_encoding_2",
        "fpn_position_encoding_3",
    };
    return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 8);
}

LanguageFeatures encode_language(Ort::Session& session, const Ort::MemoryInfo& memory, const std::string& word)
{
    const Tokens tokens = tokenize(word);
    const int64_t shape[] = {1, kContextLength};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, const_cast<int64_t*>(tokens.ids.data()), tokens.ids.size(), shape, 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory,
        const_cast<int64_t*>(tokens.attention.data()),
        tokens.attention.size(),
        shape,
        2));

    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"text_features"};
    LanguageFeatures features;
    features.attention = tokens.attention;
    features.values = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
    return features;
}

std::vector<Prediction> decode_all(
    Ort::Session& session,
    const Ort::MemoryInfo& memory,
    std::vector<Ort::Value>& image_features,
    LanguageFeatures& language,
    int image_width,
    int image_height)
{
    const int64_t mask_shape[] = {1, kContextLength};
    std::vector<Ort::Value> inputs;
    inputs.push_back(view_float(memory, image_features[0]));
    inputs.push_back(view_float(memory, image_features[1]));
    inputs.push_back(view_float(memory, image_features[2]));
    inputs.push_back(view_float(memory, image_features[6]));
    inputs.push_back(view_float(memory, language.values[0]));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory,
        language.attention.data(),
        language.attention.size(),
        mask_shape,
        2));

    const char* input_names[] = {
        "fpn_hidden_state_0",
        "fpn_hidden_state_1",
        "fpn_hidden_state_2",
        "fpn_position_encoding_2",
        "text_features",
        "attention_mask",
    };
    const char* output_names[] = {"pred_masks", "pred_boxes", "pred_logits"};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        3);

    const auto mask_info = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto box_info = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (mask_info.size() != 4 || box_info.size() != 3) {
        throw std::runtime_error("디코더 출력 형식이 예상과 다릅니다.");
    }
    const int64_t count = mask_info[1];
    const int mask_height = static_cast<int>(mask_info[2]);
    const int mask_width = static_cast<int>(mask_info[3]);
    const float* masks = outputs[0].GetTensorData<float>();
    const float* boxes = outputs[1].GetTensorData<float>();
    const float* logits = outputs[2].GetTensorData<float>();

    std::vector<Prediction> predictions;
    for (int64_t i = 0; i < count; ++i) {
        const float score = positive_score(logits[i]);
        if (!(score > 0.5f)) {
            continue;
        }

        const size_t offset = static_cast<size_t>(i) * mask_height * mask_width;
        cv::Mat low_res(mask_height, mask_width, CV_32FC1, const_cast<float*>(masks + offset));
        cv::Mat resized;
        cv::resize(low_res, resized, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

        Prediction prediction;
        prediction.score = score;
        prediction.x1 = boxes[i * 4 + 0] * static_cast<float>(image_width);
        prediction.y1 = boxes[i * 4 + 1] * static_cast<float>(image_height);
        prediction.x2 = boxes[i * 4 + 2] * static_cast<float>(image_width);
        prediction.y2 = boxes[i * 4 + 3] * static_cast<float>(image_height);
        cv::compare(resized, 0.f, prediction.mask, cv::CMP_GT);
        if (cv::countNonZero(prediction.mask) == 0) {
            continue;
        }
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

void load()
{
    tokenizer();
    image_session = open_session(kModelDir / "vision_encoder_int4.onnx", gpu_options());
    language_session = open_session(kModelDir / "text_encoder_int4.onnx", cpu_options());
    decoder_session = open_session(kModelDir / "decoder_int4.onnx", cpu_options());
}

SamOutput detect(const SamInput& input)
{
    const std::string tool = normalized_tool(input.tool);
    if (!image_session || !language_session || !decoder_session) {
        throw std::runtime_error("SAM3 Text 모델을 먼저 불러와야 합니다.");
    }
    const cv::Mat rgb = input_rgb(input);
    const ImageSize size = image_size(input.image);

    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto image_features = encode_image(*image_session, memory, rgb);
    auto language_features = encode_language(*language_session, memory, input.word);
    const auto predictions = decode_all(
        *decoder_session,
        memory,
        image_features,
        language_features,
        size.width,
        size.height);
    return to_output(predictions, tool);
}

}  // namespace sam3_text
