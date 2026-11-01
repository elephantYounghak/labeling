#pragma once

#include <filesystem>
#include <string>

std::filesystem::path utf8_to_path(const std::string& text);
std::string path_to_utf8(const std::filesystem::path& path);
