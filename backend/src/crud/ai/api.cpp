#include "crud/ai/api.h"

#include "crud/ai/core.h"
#include "crud/ai/core/rfdetr.h"
#include "utils/dataset.h"
#include "utils/path.h"

#include <drogon/MultiPart.h>
#include <drogon/drogon.h>
#include <atomic>
#include <cctype>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

drogon::HttpResponsePtr json_error(const std::string& message, drogon::HttpStatusCode code)
{
    Json::Value failed;
    failed["error"] = message;
    auto resp = drogon::HttpResponse::newHttpJsonResponse(failed);
    resp->setStatusCode(code);
    return resp;
}

bool ends_with_onnx(std::string name)
{
    for (char& letter : name) {
        letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
    }
    return name.size() >= 5 && name.compare(name.size() - 5, 5, ".onnx") == 0;
}

bool allowed_size(const std::string& size)
{
    return size == "small" || size == "medium" || size == "large";
}

int read_count(const std::string& text)
{
    if (text.empty()) {
        return 0;
    }
    try {
        std::size_t used = 0;
        const int value = std::stoi(text, &used);
        if (used != text.size()) {
            return 0;
        }
        return value;
    } catch (const std::exception&) {
        return 0;
    }
}

}  // namespace

void load_ai_api()
{
    drogon::app().registerHandler(
        "/ai/load",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            const std::string model = body ? (*body)["model"].asString() : "";

            Json::Value result(Json::objectValue);
            try {
                if (model == "sam3_text") {
                    load_sam_text();
                } else if (model == "sam3_point") {
                    load_sam_point();
                } else if (model == "rfdetr") {
                    result["name"] = rfdetr::load();
                } else {
                    throw std::runtime_error("지원하지 않는 모델입니다.");
                }
            } catch (const std::exception& error) {
                Json::Value failed;
                failed["error"] = error.what();
                auto resp = drogon::HttpResponse::newHttpJsonResponse(failed);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }

            callback(drogon::HttpResponse::newHttpJsonResponse(result));
        },
        {drogon::Post});
}

void detect_ai_api()
{
    drogon::app().registerHandler(
        "/ai/detect",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            drogon::MultiPartParser parser;
            if (parser.parse(req) != 0 || parser.getFiles().empty()) {
                Json::Value failed;
                failed["error"] = "이미지 파일이 없습니다.";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(failed);
                resp->setStatusCode(drogon::k400BadRequest);
                callback(resp);
                return;
            }

            const auto& file = parser.getFiles().front();
            Json::Value result(Json::objectValue);
            result["model"] = parser.getParameter<std::string>("model");
            result["word"] = parser.getParameter<std::string>("word");
            result["x"] = parser.getParameter<double>("x");
            result["y"] = parser.getParameter<double>("y");
            result["bytes"] = static_cast<Json::UInt64>(file.fileLength());
            try {
                const std::string model = result["model"].asString();
                if (model == "sam3_text" || model == "sam3_point") {
                    const SamInput input = make_sam_input(
                        file.fileContent(),
                        static_cast<float>(result["x"].asFloat()),
                        static_cast<float>(result["y"].asFloat()),
                        parser.getParameter<std::string>("tool"),
                        result["word"].asString());
                    const Output output = model == "sam3_text" ? detect_sam_text(input) : detect_sam_point(input);
                    const Json::Value shapes = sam_output_json(output);
                    result["boxes"] = shapes["boxes"];
                    result["masks"] = shapes["masks"];
                } else if (result["model"] == "rfdetr") {
                    rfdetr::detect();
                } else {
                    throw std::runtime_error("지원하지 않는 모델입니다.");
                }
            } catch (const std::exception& error) {
                Json::Value failed;
                failed["error"] = error.what();
                auto resp = drogon::HttpResponse::newHttpJsonResponse(failed);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }
            callback(drogon::HttpResponse::newHttpJsonResponse(result));
        },
        {drogon::Post});
        drogon::app().registerHandler(
            "/ai/detect-all",
            [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
                drogon::MultiPartParser parser;
                if (parser.parse(req) != 0 || parser.getFiles().empty()) {
                    Json::Value failed;
                    failed["error"] = "이미지 파일이 없습니다.";
                    auto resp = drogon::HttpResponse::newHttpJsonResponse(failed);
                    resp->setStatusCode(drogon::k400BadRequest);
                    callback(resp);
                    return;
                }
            },
            {drogon::Post});
}

void upload_ai_api()
{
    drogon::app().registerHandler(
        "/ai/upload",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            drogon::MultiPartParser parser;
            if (parser.parse(req) != 0 || parser.getFiles().empty()) {
                callback(json_error("ONNX 파일이 없습니다.", drogon::k400BadRequest));
                return;
            }
            const auto& file = parser.getFiles().front();
            if (!ends_with_onnx(file.getFileName())) {
                callback(json_error("ONNX 파일만 불러올 수 있습니다.", drogon::k400BadRequest));
                return;
            }
            try {
                Json::Value saved;
                saved["name"] = rfdetr::upload(file.fileContent(), file.getFileName());
                callback(drogon::HttpResponse::newHttpJsonResponse(saved));
            } catch (const std::exception& error) {
                callback(json_error(error.what(), drogon::k500InternalServerError));
            }
        },
        {drogon::Post});
}

void train_ai_api()
{
    drogon::app().registerHandler(
        "/ai/train",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            drogon::MultiPartParser parser;
            if (parser.parse(req) != 0) {
                callback(json_error("학습 파일을 읽지 못했습니다.", drogon::k400BadRequest));
                return;
            }
            const std::string size = parser.getParameter<std::string>("size");
            const std::string tool = parser.getParameter<std::string>("tool");
            const int epochs = read_count(parser.getParameter<std::string>("epochs"));
            const int batch_size = read_count(parser.getParameter<std::string>("batchSize"));
            if (!allowed_size(size)) {
                callback(json_error("모델 크기는 small, medium, large 중에서 고르세요.", drogon::k400BadRequest));
                return;
            }
            if (tool != "bbox" && tool != "polygon" && tool != "brush") {
                callback(json_error("라벨 도구는 bbox, polygon, brush 중에서 고르세요.", drogon::k400BadRequest));
                return;
            }
            if (epochs < 1) {
                callback(json_error("epoch은 1 이상이어야 합니다.", drogon::k400BadRequest));
                return;
            }
            if (batch_size < 1) {
                callback(json_error("batch size는 1 이상이어야 합니다.", drogon::k400BadRequest));
                return;
            }

            const drogon::HttpFile* annotation = nullptr;
            std::vector<const drogon::HttpFile*> images;
            for (const drogon::HttpFile& file : parser.getFiles()) {
                if (file.getItemName() == "annotation") {
                    annotation = &file;
                } else if (file.getItemName() == "images") {
                    images.push_back(&file);
                }
            }
            if (!annotation) {
                callback(json_error("라벨 JSON이 없습니다.", drogon::k400BadRequest));
                return;
            }
            if (images.empty()) {
                callback(json_error("라벨된 이미지가 없습니다.", drogon::k400BadRequest));
                return;
            }

            Json::Value paths;
            Json::CharReaderBuilder builder;
            const std::string paths_text = parser.getParameter<std::string>("paths");
            std::string errors;
            const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            if (!reader->parse(paths_text.data(), paths_text.data() + paths_text.size(), &paths, &errors) || !paths.isArray()) {
                callback(json_error("이미지 경로가 없습니다.", drogon::k400BadRequest));
                return;
            }

            static std::atomic<int> next_id{1};
            const std::string id = std::to_string(next_id.fetch_add(1));
            Json::Value saved;
            saved["id"] = id;
            try {
                std::vector<std::string_view> image_bytes;
                image_bytes.reserve(images.size());
                for (const drogon::HttpFile* image : images) {
                    image_bytes.push_back(image->fileContent());
                }
                const std::filesystem::path dataset = save_train_files(id, annotation->fileContent(), image_bytes, paths);
                rfdetr::train(path_to_utf8(dataset), size, tool, epochs, batch_size);
                callback(drogon::HttpResponse::newHttpJsonResponse(saved));
            } catch (const std::exception& error) {
                callback(json_error(error.what(), drogon::k500InternalServerError));
            }
        },
        {drogon::Post});
}
