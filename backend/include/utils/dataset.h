#pragma once

#include <json/json.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

std::filesystem::path save_train_files(const std::string& id, std::string_view annotation,
    const std::vector<std::string_view>& images, const Json::Value& paths);
