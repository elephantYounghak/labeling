#include "crud/ai/api.h"

#include "crud/ai/core.h"

#include <drogon/MultiPart.h>
#include <drogon/drogon.h>
#include <atomic>
#include <cctype>
#include <exception>
#include <string>

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
                    result["test"] = load_rfdetr();
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
                    const SamOutput output = model == "sam3_text" ? detect_sam_text(input) : detect_sam_point(input);
                    const Json::Value shapes = sam_output_json(output);
                    result["boxes"] = shapes["boxes"];
                    result["masks"] = shapes["masks"];
                } else if (result["model"] == "rfdetr") {
                    detect_rfdetr();
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
            Json::Value saved;
            saved["name"] = file.getFileName();
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Post});
}

void train_ai_api()
{
    drogon::app().registerHandler(
        "/ai/train",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            const std::string project_id = body ? (*body)["projectId"].asString() : "";
            const std::string size = body ? (*body)["size"].asString() : "";
            const int epochs = body ? (*body)["epochs"].asInt() : 0;
            if (project_id.empty()) {
                callback(json_error("프로젝트 번호가 없습니다.", drogon::k400BadRequest));
                return;
            }
            if (!allowed_size(size)) {
                callback(json_error("모델 크기는 small, medium, large 중에서 고르세요.", drogon::k400BadRequest));
                return;
            }
            if (epochs < 1) {
                callback(json_error("epoch은 1 이상이어야 합니다.", drogon::k400BadRequest));
                return;
            }

            static std::atomic<int> next_id{1};
            Json::Value saved;
            saved["id"] = std::to_string(next_id.fetch_add(1));
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Post});
}
