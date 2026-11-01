#pragma once

#include <json/json.h>

#include <filesystem>
#include <string>
#include <vector>

std::vector<std::string> coco_classes(const Json::Value& root);
void write_coco(const std::filesystem::path& path, const Json::Value& split, const std::filesystem::path& image_root,
    const std::vector<std::string>& classes);
