#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

struct RfdetrBox
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
    std::string name;
};

struct RfdetrMask
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float score = 0;
    std::string name;
};

struct RfdetrOutput
{
    std::vector<RfdetrBox> boxes;
};

using RfdetrOutputs = std::map<std::string, RfdetrOutput>;

namespace rfdetr {

std::string load();
RfdetrOutput detect();
std::string upload(std::string_view bytes, const std::string& filename);
void train(const std::string& dataset_dir, const std::string& size, const std::string& tool, int epochs, int batch_size);

}  // namespace rfdetr
