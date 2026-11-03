#pragma once

#include "crud/ai/core/sam3.h"

#include <json/json.h>

#include <string>
#include <string_view>

void load_sam_text();
void load_sam_point();
SamInput make_sam_input(std::string_view bytes, float x, float y, const std::string& tool, const std::string& word);
SamOutput detect_sam_text(const SamInput& input);
SamOutput detect_sam_point(const SamInput& input);
Json::Value sam_output_json(const SamOutput& output);

std::string load();
std::string detect();
std::string upload(std::string_view bytes, const std::string& filename);
void train_rfdetr(const std::string& dataset_dir, const std::string& size, int epochs, int batch_size);
