#include "utils/coco.h"

#include "utils/image.h"
#include "utils/path.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

std::string image_name(const Json::Value& row)
{
    if (!row.isObject()) {
        return {};
    }
    if (row["image"].isString()) {
        return row["image"].asString();
    }
    if (row["file"].isString()) {
        return row["file"].asString();
    }
    return {};
}

bool normalized_box(const Json::Value& row, double& x, double& y, double& width, double& height)
{
    const Json::Value& mask = row["mask"];
    if (mask.isObject() && mask["box"].isObject()) {
        const Json::Value& box = mask["box"];
        if (box["x"].isNumeric() && box["y"].isNumeric() && box["w"].isNumeric() && box["h"].isNumeric()) {
            x = box["x"].asDouble();
            y = box["y"].asDouble();
            width = box["w"].asDouble();
            height = box["h"].asDouble();
            return width > 0.0 && height > 0.0;
        }
    }

    const Json::Value& points = row["points"];
    if (!points.isArray() || points.empty()) {
        return false;
    }
    double min_x = 1.0;
    double min_y = 1.0;
    double max_x = 0.0;
    double max_y = 0.0;
    bool any = false;
    for (const Json::Value& point : points) {
        if (!point.isObject() || !point["x"].isNumeric() || !point["y"].isNumeric()) {
            continue;
        }
        any = true;
        min_x = (std::min)(min_x, point["x"].asDouble());
        min_y = (std::min)(min_y, point["y"].asDouble());
        max_x = (std::max)(max_x, point["x"].asDouble());
        max_y = (std::max)(max_y, point["y"].asDouble());
    }
    if (!any) {
        return false;
    }
    x = min_x;
    y = min_y;
    width = max_x - min_x;
    height = max_y - min_y;
    return width > 0.0 && height > 0.0;
}

int class_id(const std::vector<std::string>& names, const Json::Value& row)
{
    std::string name = row["name"].isString() ? row["name"].asString() : "object";
    if (name.empty()) {
        name = "object";
    }
    const auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end()) {
        return 0;
    }
    return static_cast<int>(found - names.begin());
}

ImageSize image_pixel_size(const fs::path& image_file, const Json::Value& rows, const std::string& image)
{
    for (const Json::Value& row : rows) {
        if (image_name(row) != image || !row["mask"].isObject()) {
            continue;
        }
        const Json::Value& mask = row["mask"];
        if (mask["width"].isNumeric() && mask["height"].isNumeric() && mask["width"].asInt() > 0 && mask["height"].asInt() > 0) {
            return {mask["width"].asInt(), mask["height"].asInt()};
        }
    }

    std::ifstream input(image_file, std::ios::binary);
    if (!input) {
        throw std::runtime_error("라벨 이미지를 열지 못했습니다.");
    }
    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return image_size(image_from_bytes(bytes));
}

Json::Value make_coco(const Json::Value& split, const fs::path& split_root, const std::vector<std::string>& classes)
{
    const Json::Value& rows = split["annotations"];
    std::vector<std::string> images;
    for (const Json::Value& row : rows) {
        const std::string image = image_name(row);
        if (!image.empty() && std::find(images.begin(), images.end(), image) == images.end()) {
            images.push_back(image);
        }
    }

    Json::Value coco(Json::objectValue);
    coco["info"] = Json::Value(Json::objectValue);
    coco["categories"] = Json::Value(Json::arrayValue);
    for (int index = 0; index < static_cast<int>(classes.size()); ++index) {
        Json::Value category(Json::objectValue);
        category["id"] = index;
        category["name"] = classes[static_cast<size_t>(index)];
        coco["categories"].append(std::move(category));
    }

    coco["images"] = Json::Value(Json::arrayValue);
    std::vector<ImageSize> sizes;
    sizes.reserve(images.size());
    for (int index = 0; index < static_cast<int>(images.size()); ++index) {
        const fs::path image_file = split_root / utf8_to_path(images[static_cast<size_t>(index)]);
        const ImageSize size = image_pixel_size(image_file, rows, images[static_cast<size_t>(index)]);
        sizes.push_back(size);
        Json::Value image(Json::objectValue);
        image["id"] = index;
        image["file_name"] = images[static_cast<size_t>(index)];
        image["width"] = size.width;
        image["height"] = size.height;
        coco["images"].append(std::move(image));
    }

    coco["annotations"] = Json::Value(Json::arrayValue);
    int annotation_id = 0;
    for (const Json::Value& row : rows) {
        const std::string image = image_name(row);
        const auto found = std::find(images.begin(), images.end(), image);
        if (found == images.end()) {
            continue;
        }
        double x = 0.0;
        double y = 0.0;
        double width = 0.0;
        double height = 0.0;
        if (!normalized_box(row, x, y, width, height)) {
            continue;
        }
        const int image_id = static_cast<int>(found - images.begin());
        const ImageSize size = sizes[static_cast<size_t>(image_id)];
        const double x1 = std::clamp(x, 0.0, 1.0) * size.width;
        const double y1 = std::clamp(y, 0.0, 1.0) * size.height;
        const double x2 = std::clamp(x + width, 0.0, 1.0) * size.width;
        const double y2 = std::clamp(y + height, 0.0, 1.0) * size.height;
        const double box_w = (std::max)(0.0, x2 - x1);
        const double box_h = (std::max)(0.0, y2 - y1);
        if (box_w <= 0.0 || box_h <= 0.0) {
            continue;
        }

        Json::Value annotation(Json::objectValue);
        annotation["id"] = annotation_id;
        annotation["image_id"] = image_id;
        annotation["category_id"] = class_id(classes, row);
        annotation["bbox"] = Json::Value(Json::arrayValue);
        annotation["bbox"].append(x1);
        annotation["bbox"].append(y1);
        annotation["bbox"].append(box_w);
        annotation["bbox"].append(box_h);
        annotation["area"] = box_w * box_h;
        annotation["iscrowd"] = 0;
        annotation["ignore"] = 0;

        const Json::Value& mask = row["mask"];
        if (mask.isObject() && mask["counts"].isString() && !mask["counts"].asString().empty()) {
            Json::Value segmentation(Json::objectValue);
            segmentation["size"] = Json::Value(Json::arrayValue);
            segmentation["size"].append(size.height);
            segmentation["size"].append(size.width);
            segmentation["counts"] = mask["counts"].asString();
            annotation["segmentation"] = std::move(segmentation);
        } else {
            Json::Value ring(Json::arrayValue);
            ring.append(x1);
            ring.append(y1);
            ring.append(x2);
            ring.append(y1);
            ring.append(x2);
            ring.append(y2);
            ring.append(x1);
            ring.append(y2);
            Json::Value segmentation(Json::arrayValue);
            segmentation.append(std::move(ring));
            annotation["segmentation"] = std::move(segmentation);
        }
        coco["annotations"].append(std::move(annotation));
        ++annotation_id;
    }
    return coco;
}

}  // namespace

std::vector<std::string> coco_classes(const Json::Value& root)
{
    std::vector<std::string> names;
    for (const Json::Value& row : root["annotations"]) {
        std::string name = row["name"].isString() ? row["name"].asString() : "object";
        if (name.empty()) {
            name = "object";
        }
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }
    if (names.empty()) {
        names.push_back("object");
    }
    return names;
}

void write_coco(const fs::path& path, const Json::Value& split, const fs::path& image_root, const std::vector<std::string>& classes)
{
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("COCO 라벨을 만들지 못했습니다.");
    }
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(make_coco(split, image_root, classes), &output);
    output << '\n';
    if (!output) {
        throw std::runtime_error("COCO 라벨을 만들지 못했습니다.");
    }
}
