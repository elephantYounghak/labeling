#pragma once

#include <string>

class Project
{
public:
    Project() = default;

    std::string id;
    std::string name;
    std::string description;
    std::string path;
    std::string created_at;
    std::string updated_at;
};

void upload_project(Project& project);
