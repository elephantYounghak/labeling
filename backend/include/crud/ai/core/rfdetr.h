#pragma once

#include "crud/ai/core.h"

#include <string>
#include <string_view>

namespace rfdetr {

void load();
Output detect();
std::string upload(std::string_view bytes, const std::string& filename);
void train(const std::string& dataset_dir, const std::string& size, const std::string& tool, int epochs, int batch_size);

}  // namespace rfdetr
