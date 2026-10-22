#pragma once

#include <json/json.h>

#include <string>

bool json_exists(const std::string& folder);
Json::Value read_json(const std::string& folder);
Json::Value create_json(const std::string& folder);
