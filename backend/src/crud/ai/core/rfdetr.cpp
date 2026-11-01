#include "crud/ai/core.h"

#include "utils/coco.h"
#include "utils/path.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::string load_rfdetr()
{
    return "test";
}

std::string detect_rfdetr()
{
    return "test";
}

std::string make_train_command(const std::string& dataset_dir, const std::string& size,
    int epochs, int batch_size)
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


void train_rfdetr(const std::string& dataset_dir, const std::string& size, 
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

    std::string command = make_train_command(dataset_dir, size, epochs, batch_size);
    if (system(command.c_str()) != 0) {
        throw std::runtime_error("학습 스크립트가 실패했습니다.");
    }
}