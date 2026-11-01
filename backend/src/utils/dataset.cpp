#include "utils/dataset.h"

#include "utils/path.h"

#include <fstream>
#include <stdexcept>

namespace {

bool safe_relative_path(const std::string& text)
{
    if (text.empty() || text.find(':') != std::string::npos || text.front() == '/' || text.front() == '\\') {
        return false;
    }
    const std::filesystem::path relative = utf8_to_path(text);
    if (relative.empty() || relative.is_absolute()) {
        return false;
    }
    for (const std::filesystem::path& part : relative) {
        if (part.empty() || part == "." || part == "..") {
            return false;
        }
        if (part.wstring().find_first_of(L"<>:\"|?*") != std::wstring::npos) {
            return false;
        }
    }
    return true;
}

void write_bytes(const std::filesystem::path& path, std::string_view bytes)
{
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("학습 파일을 저장하지 못했습니다.");
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("학습 파일을 저장하지 못했습니다.");
    }
}

}  // namespace

std::filesystem::path save_train_files(const std::string& id, std::string_view annotation,
    const std::vector<std::string_view>& images, const Json::Value& paths)
{
    if (!paths.isArray() || paths.size() != images.size()) {
        throw std::runtime_error("이미지 경로가 파일 개수와 맞지 않습니다.");
    }
    const std::filesystem::path root = std::filesystem::path("backend/data/datasets") / id;
    write_bytes(root / L"_annotation_.json", annotation);
    for (Json::ArrayIndex index = 0; index < paths.size(); ++index) {
        if (!paths[index].isString()) {
            throw std::runtime_error("이미지 경로가 없습니다.");
        }
        const std::string relative = paths[index].asString();
        if (!safe_relative_path(relative)) {
            throw std::runtime_error("이미지 경로가 폴더 밖을 가리킵니다.");
        }
        write_bytes(root / utf8_to_path(relative), images[index]);
    }
    return root;
}
