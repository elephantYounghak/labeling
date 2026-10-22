#pragma once

#include <string>

struct Labeling
{
    std::string project_id;
    std::string tool;
    double opacity = 0.4;
    std::string classes;
};

void upload_labeling(Labeling& labeling);
void update_labeling(Labeling& labeling);
