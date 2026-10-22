#include "crud/io/core.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

constexpr const wchar_t* kFileName = L"_annotation_.json";

std::filesystem::path utf8_to_path(const std::string& text)
{
    if (text.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 1) {
        return {};
    }
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    return std::filesystem::path(wide);
}

std::filesystem::path annotation_file(const std::string& folder)
{
    if (folder.empty()) {
        throw std::runtime_error("폴더 경로가 비어 있습니다.");
    }
    const std::filesystem::path directory = utf8_to_path(folder);
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("폴더를 찾지 못했습니다.");
    }
    return directory / kFileName;
}

}  // namespace

bool json_exists(const std::string& folder)
{
    try {
        return std::filesystem::is_regular_file(annotation_file(folder));
    } catch (const std::runtime_error&) {
        return false;
    }
}

Json::Value read_json(const std::string& folder)
{
    std::ifstream input(annotation_file(folder));
    if (!input) {
        throw std::runtime_error("JSON 파일을 열지 못했습니다.");
    }
    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errors;
    if (!Json::parseFromStream(builder, input, &root, &errors)) {
        throw std::runtime_error(errors.empty() ? "JSON 형식이 아닙니다." : errors);
    }
    return root;
}

Json::Value create_json(const std::string& folder)
{
    const std::filesystem::path file = annotation_file(folder);
    Json::Value root(Json::objectValue);
    root["annotations"] = Json::Value(Json::arrayValue);

    std::ofstream output(file);
    if (!output) {
        throw std::runtime_error("JSON 파일을 만들지 못했습니다.");
    }
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &output);
    output << '\n';
    return root;
}
