#include "crud/ai/core/rfdetr.h"
#include "crud/ai/core/sam3.h"

#include "utils/coco.h"
#include "utils/path.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string image_name(const Json::Value& row)
{
    if (!row.isObject()) {
        return {};
    }
    if (row["image"].isString()) {
        return row["image"].asString();
    }
    if (row["file"].isString()) {
        return row["file"].asString();
    }
    return {};
}

}  // namespace

namespace {

const fs::path kModelDir = "backend/data/models/rfdetr";
std::unique_ptr<Ort::Session> session;

bool ends_with_onnx(std::string name)
{
    for (char& letter : name) {
        letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
    }
    return name.size() >= 5 && name.compare(name.size() - 5, 5, ".onnx") == 0;
}

std::string onnx_file_name(const std::string& filename)
{
    const std::string name = path_to_utf8(utf8_to_path(filename).filename());
    if (name.empty() || name == "." || name == ".." || !ends_with_onnx(name)) {
        throw std::runtime_error("ONNX 파일만 불러올 수 있습니다.");
    }
    if (name.find_first_of("<>:\"|?*") != std::string::npos) {
        throw std::runtime_error("ONNX 파일 이름을 쓰지 못했습니다.");
    }
    return name;
}

fs::path newest_onnx()
{
    if (!fs::is_directory(kModelDir)) {
        throw std::runtime_error("RF-DETR ONNX 파일이 없습니다. 추론 탭에서 모델을 먼저 불러오세요.");
    }
    fs::path best;
    fs::file_time_type newest{};
    bool found = false;
    for (const fs::directory_entry& entry : fs::directory_iterator(kModelDir)) {
        if (!entry.is_regular_file() || !ends_with_onnx(path_to_utf8(entry.path().filename()))) {
            continue;
        }
        const fs::file_time_type written = entry.last_write_time();
        if (!found || written > newest) {
            found = true;
            newest = written;
            best = entry.path();
        }
    }
    if (!found) {
        throw std::runtime_error("RF-DETR ONNX 파일이 없습니다. 추론 탭에서 모델을 먼저 불러오세요.");
    }
    return best;
}

constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};

int positive_dim(int64_t value)
{
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("RF-DETR 텐서 크기를 읽지 못했습니다.");
    }
    return static_cast<int>(value);
}

std::vector<float> imagenet_chw(const cv::Mat& bgr, int width, int height)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    std::vector<float> pixels(static_cast<size_t>(3) * width * height);
    for (int y = 0; y < height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = static_cast<float>(row[x][channel]) / 255.f;
                pixels[static_cast<size_t>(channel) * width * height + static_cast<size_t>(y) * width + x]
                    = (value - kImageNetMean[channel]) / kImageNetStd[channel];
            }
        }
    }
    return pixels;
}

float logit_score(float logit)
{
    const float clipped = std::clamp(logit, -88.f, 88.f);
    return 1.f / (1.f + std::exp(-clipped));
}

struct Detection
{
    float score = 0;
    int query = 0;
    int class_id = 0;
    int order = 0;
};

std::vector<Detection> select_detections(const float* logits, int queries, int classes, float threshold)
{
    const int foreground = classes > 1 ? classes - 1 : classes;
    std::vector<Detection> pairs;
    pairs.reserve(static_cast<size_t>(queries) * static_cast<size_t>(foreground));
    int order = 0;
    for (int query = 0; query < queries; ++query) {
        for (int cls = 0; cls < foreground; ++cls) {
            pairs.push_back({logit_score(logits[query * classes + cls]), query, cls, order});
            ++order;
        }
    }
    if (pairs.empty()) {
        return {};
    }

    const auto keep_end = pairs.begin() + std::min(queries, static_cast<int>(pairs.size()));
    std::partial_sort(pairs.begin(), keep_end, pairs.end(), [](const Detection& left, const Detection& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return left.order < right.order;
    });
    pairs.erase(keep_end, pairs.end());

    std::vector<Detection> kept;
    for (const Detection& detection : pairs) {
        if (detection.score > threshold) {
            kept.push_back(detection);
        }
    }
    return kept;
}

std::vector<int64_t> tensor_shape(const Ort::Value& value)
{
    return value.GetTensorTypeAndShapeInfo().GetShape();
}

const float* tensor_floats(const Ort::Value& value)
{
    if (value.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("RF-DETR 출력 형식을 읽지 못했습니다.");
    }
    return value.GetTensorData<float>();
}

std::vector<int64_t> drop_batch(const std::vector<int64_t>& shape, int rank)
{
    if (static_cast<int>(shape.size()) != rank || shape[0] != 1) {
        throw std::runtime_error("RF-DETR 출력 형식을 읽지 못했습니다.");
    }
    return {shape.begin() + 1, shape.end()};
}

struct ModelOutputs
{
    int boxes = -1;
    int logits = -1;
    int masks = -1;
};

ModelOutputs match_outputs(Ort::Session& model)
{
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = model.GetOutputCount();
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    names.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        names.push_back(model.GetOutputNameAllocated(index, allocator).get());
        shapes.push_back(model.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape());
    }

    ModelOutputs found;
    for (int index = 0; index < static_cast<int>(names.size()); ++index) {
        const std::string& name = names[index];
        if (name.find("dets") != std::string::npos) {
            found.boxes = index;
        } else if (name.find("labels") != std::string::npos) {
            found.logits = index;
        } else if (name.find("masks") != std::string::npos) {
            found.masks = index;
        }
    }
    if (found.boxes >= 0 && found.logits >= 0) {
        return found;
    }

    std::vector<int> boxes;
    std::vector<int> logits;
    std::vector<int> rank3;
    for (int index = 0; index < static_cast<int>(shapes.size()); ++index) {
        const std::vector<int64_t>& shape = shapes[index];
        if (shape.size() == 4 && found.masks < 0) {
            found.masks = index;
        }
        if (shape.size() != 3) {
            continue;
        }
        rank3.push_back(index);
        if (shape.back() == 4) {
            boxes.push_back(index);
        } else {
            logits.push_back(index);
        }
    }
    if (boxes.size() == 1 && logits.size() == 1) {
        found.boxes = boxes[0];
        found.logits = logits[0];
        return found;
    }
    if (rank3.size() >= 2) {
        found.boxes = rank3[0];
        found.logits = rank3[1];
        return found;
    }
    throw std::runtime_error("RF-DETR 출력 형식을 읽지 못했습니다.");
}

std::vector<Point> mask_points(const float* logits, int mask_height, int mask_width, int image_width, int image_height)
{
    cv::Mat low(mask_height, mask_width, CV_32FC1, const_cast<float*>(logits));
    cv::Mat resized;
    cv::resize(low, resized, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat binary;
    cv::compare(resized, 0.f, binary, cv::CMP_GT);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }

    size_t best = 0;
    double best_area = 0;
    for (size_t index = 0; index < contours.size(); ++index) {
        const double area = cv::contourArea(contours[index]);
        if (area > best_area) {
            best_area = area;
            best = index;
        }
    }

    std::vector<cv::Point> shape = contours[best];
    std::vector<cv::Point> approx;
    cv::approxPolyDP(shape, approx, 0.01 * cv::arcLength(shape, true), true);
    if (approx.size() >= 3) {
        shape = std::move(approx);
    }
    if (shape.size() < 3) {
        return {};
    }

    std::vector<Point> points;
    points.reserve(shape.size());
    for (const cv::Point& point : shape) {
        points.push_back({static_cast<float>(point.x), static_cast<float>(point.y)});
    }
    return points;
}

}  // namespace

Output rfdetr::detect(const RfdetrInput& input)
{
    if (!session) {
        throw std::runtime_error("RF-DETR 모델을 먼저 불러와야 합니다.");
    }
    if (input.image.empty()) {
        throw std::runtime_error("이미지가 없습니다.");
    }

    const auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("RF-DETR 입력 형식을 읽지 못했습니다.");
    }
    if (input_shape[1] != 3) {
        throw std::runtime_error("RF-DETR 입력 형식을 읽지 못했습니다.");
    }
    const int height = positive_dim(input_shape[2]);
    const int width = positive_dim(input_shape[3]);
    std::vector<float> pixels = imagenet_chw(input.image, width, height);
    const int64_t input_dims[] = {1, 3, height, width};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory, pixels.data(), pixels.size(), input_dims, 4);

    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(0, allocator);
    std::vector<Ort::AllocatedStringPtr> output_name_owners;
    std::vector<const char*> output_names;
    output_name_owners.reserve(session->GetOutputCount());
    for (size_t index = 0; index < session->GetOutputCount(); ++index) {
        output_name_owners.push_back(session->GetOutputNameAllocated(index, allocator));
        output_names.push_back(output_name_owners.back().get());
    }
    const char* input_names[] = {input_name.get()};
    const ModelOutputs selected = match_outputs(*session);
    auto values = session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &tensor,
        1,
        output_names.data(),
        output_names.size());

    const std::vector<int64_t> box_shape = drop_batch(tensor_shape(values[selected.boxes]), 3);
    const std::vector<int64_t> logit_shape = drop_batch(tensor_shape(values[selected.logits]), 3);
    if (box_shape.size() != 2 || box_shape[1] != 4 || logit_shape.size() != 2 || box_shape[0] != logit_shape[0]) {
        throw std::runtime_error("RF-DETR 출력 형식을 읽지 못했습니다.");
    }
    const int queries = positive_dim(box_shape[0]);
    const int classes = positive_dim(logit_shape[1]);
    const float* boxes = tensor_floats(values[selected.boxes]);
    const float* logits = tensor_floats(values[selected.logits]);
    const float threshold = std::clamp(input.threshold, 0.f, 1.f);
    const std::vector<Detection> detections = select_detections(logits, queries, classes, threshold);

    const float* masks = nullptr;
    int mask_height = 0;
    int mask_width = 0;
    if (selected.masks >= 0) {
        const std::vector<int64_t> mask_shape = drop_batch(tensor_shape(values[selected.masks]), 4);
        if (mask_shape.size() != 3 || mask_shape[0] != queries) {
            throw std::runtime_error("RF-DETR 출력 형식을 읽지 못했습니다.");
        }
        mask_height = positive_dim(mask_shape[1]);
        mask_width = positive_dim(mask_shape[2]);
        masks = tensor_floats(values[selected.masks]);
    }

    const float image_width = static_cast<float>(input.image.cols);
    const float image_height = static_cast<float>(input.image.rows);
    Output output;
    for (const Detection& detection : detections) {
        const float* box = boxes + static_cast<size_t>(detection.query) * 4;
        Box item;
        item.x1 = std::clamp((box[0] - box[2] / 2.f) * image_width, 0.f, image_width);
        item.y1 = std::clamp((box[1] - box[3] / 2.f) * image_height, 0.f, image_height);
        item.x2 = std::clamp((box[0] + box[2] / 2.f) * image_width, 0.f, image_width);
        item.y2 = std::clamp((box[1] + box[3] / 2.f) * image_height, 0.f, image_height);
        item.score = detection.score;
        item.class_id = detection.class_id;
        if (item.x2 <= item.x1 || item.y2 <= item.y1) {
            continue;
        }
        output.boxes.push_back(item);
        if (masks == nullptr) {
            continue;
        }
        const size_t offset = static_cast<size_t>(detection.query) * mask_height * mask_width;
        Mask mask;
        mask.score = detection.score;
        mask.class_id = detection.class_id;
        mask.points = mask_points(masks + offset, mask_height, mask_width, input.image.cols, input.image.rows);
        if (!mask.points.empty()) {
            output.masks.push_back(std::move(mask));
        }
    }
    return output;
}

void rfdetr::load()
{
    const fs::path file = newest_onnx();
    session = sam3::open_session(file, sam3::gpu_options());
}

std::string rfdetr::upload(std::string_view bytes, const std::string& filename)
{
    if (bytes.empty()) {
        throw std::runtime_error("ONNX 파일이 없습니다.");
    }
    fs::create_directories(kModelDir);
    const fs::path file = kModelDir / utf8_to_path(onnx_file_name(filename));
    std::ofstream output(file, std::ios::binary);
    if (!output) {
        throw std::runtime_error("ONNX 파일을 저장하지 못했습니다.");
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("ONNX 파일을 저장하지 못했습니다.");
    }
    return path_to_utf8(file.filename());
}

std::string make_train_command(const std::string& dataset_dir, const std::string& size,
    const std::string& tool, int epochs, int batch_size)
{
    const fs::path script = "backend/src/crud/ai/core/rfdetr.py";
    if (!fs::is_regular_file(script)) {
        throw std::runtime_error("학습 스크립트가 없습니다.");
    }

    std::string dataset = dataset_dir;
    if (!dataset.empty() && (dataset.back() == '\\' || dataset.back() == '/')) {
        dataset.pop_back();
    }

    return "uv run --project backend -- python -P \"" + script.generic_string() + "\" \"" + dataset + "\""
        + " --size " + size
        + " --tool " + tool
        + " --epochs " + std::to_string(epochs)
        + " --batch-size " + std::to_string(batch_size);
}

std::pair<Json::Value, Json::Value> split_annotation_json(const Json::Value& root)
{
    const Json::Value& rows = root["annotations"];
    if (!rows.isArray()) {
        throw std::runtime_error("라벨 JSON 형식이 아닙니다.");
    }

    std::vector<std::string> images;
    for (const Json::Value& row : rows) {
        const std::string image = image_name(row);
        if (!image.empty() && std::find(images.begin(), images.end(), image) == images.end()) {
            images.push_back(image);
        }
    }

    const int valid_count = static_cast<int>(images.size()) / 10;
    const auto valid_begin = images.end() - valid_count;
    Json::Value train = root;
    Json::Value valid = root;
    train["annotations"] = Json::Value(Json::arrayValue);
    valid["annotations"] = Json::Value(Json::arrayValue);
    for (const Json::Value& row : rows) {
        const std::string image = image_name(row);
        const bool to_valid = std::find(valid_begin, images.end(), image) != images.end();
        (to_valid ? valid["annotations"] : train["annotations"]).append(row);
    }
    return {std::move(train), std::move(valid)};
}

void copy_annotation_images(const Json::Value& root, const std::string& source_dir, const std::string& target_dir)
{
    const Json::Value& rows = root["annotations"];
    if (!rows.isArray()) {
        throw std::runtime_error("라벨 JSON 형식이 아닙니다.");
    }
    if (source_dir.empty() || target_dir.empty()) {
        throw std::runtime_error("이미지 폴더가 없습니다.");
    }
    const fs::path source_root = utf8_to_path(source_dir);
    const fs::path target_root = utf8_to_path(target_dir);
    if (!fs::is_directory(source_root)) {
        throw std::runtime_error("이미지 폴더를 찾지 못했습니다.");
    }
    fs::create_directories(target_root);

    std::vector<std::string> images;
    for (const Json::Value& row : rows) {
        const std::string image = image_name(row);
        if (image.empty() || std::find(images.begin(), images.end(), image) != images.end()) {
            continue;
        }
        images.push_back(image);
    }

    for (const std::string& image : images) {
        const fs::path relative = utf8_to_path(image);
        bool outside = relative.empty() || relative.is_absolute();
        for (const fs::path& part : relative) {
            if (part == "..") {
                outside = true;
            }
        }
        if (outside) {
            throw std::runtime_error("라벨 이미지 경로가 폴더 밖을 가리킵니다.");
        }
        const fs::path source = source_root / relative;
        if (!fs::is_regular_file(source)) {
            throw std::runtime_error("라벨 이미지를 찾지 못했습니다.");
        }
        const fs::path target = target_root / relative;
        if (target.has_parent_path()) {
            fs::create_directories(target.parent_path());
        }
        fs::copy_file(source, target, fs::copy_options::overwrite_existing);
    }
}

namespace {

void remove_uploaded_sources(const fs::path& dataset_root, const Json::Value& root)
{
    std::vector<std::string> images;
    for (const Json::Value& row : root["annotations"]) {
        const std::string image = image_name(row);
        if (!image.empty() && std::find(images.begin(), images.end(), image) == images.end()) {
            images.push_back(image);
        }
    }
    for (const std::string& image : images) {
        const fs::path file = dataset_root / utf8_to_path(image);
        if (fs::is_regular_file(file)) {
            fs::remove(file);
        }
    }
    const fs::path annotation = dataset_root / L"_annotation_.json";
    if (fs::is_regular_file(annotation)) {
        fs::remove(annotation);
    }
}

}  // namespace


void rfdetr::train(const std::string& dataset_dir, const std::string& size, const std::string& tool,
    int epochs, int batch_size)
{
    if (dataset_dir.empty()) {
        throw std::runtime_error("학습 데이터 폴더가 없습니다.");
    }
    if (epochs < 1 || batch_size < 1) {
        throw std::runtime_error("epoch과 batch size는 1 이상이어야 합니다.");
    }

    const fs::path script = "backend/src/crud/ai/core/rfdetr.py";
    if (!fs::is_regular_file(script)) {
        throw std::runtime_error("학습 스크립트가 없습니다.");
    }

    const fs::path dataset_root = utf8_to_path(dataset_dir);
    if (!fs::is_directory(dataset_root)) {
        throw std::runtime_error("학습 데이터 폴더를 찾지 못했습니다.");
    }
    std::ifstream input(dataset_root / L"_annotation_.json");
    if (!input) {
        throw std::runtime_error("라벨 JSON 파일을 열지 못했습니다.");
    }
    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errors;
    if (!Json::parseFromStream(builder, input, &root, &errors)) {
        throw std::runtime_error(errors.empty() ? "라벨 JSON 형식이 아닙니다." : errors);
    }
    input.close();

    const std::pair<Json::Value, Json::Value> split = split_annotation_json(root);
    const std::vector<std::string> classes = coco_classes(root);
    const fs::path train_dir = dataset_root / "train";
    const fs::path valid_dir = dataset_root / "valid";
    copy_annotation_images(split.first, dataset_dir, train_dir.u8string());
    copy_annotation_images(split.second, dataset_dir, valid_dir.u8string());
    write_coco(train_dir / L"_annotations.coco.json", split.first, train_dir, classes);
    write_coco(valid_dir / L"_annotations.coco.json", split.second, valid_dir, classes);
    remove_uploaded_sources(dataset_root, root);

    std::string command = make_train_command(dataset_dir, size, tool, epochs, batch_size);
    if (system(command.c_str()) != 0) {
        throw std::runtime_error("학습 스크립트가 실패했습니다.");
    }
}