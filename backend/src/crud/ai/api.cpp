#include "crud/ai/api.h"

#include "crud/ai/core.h"

#include <drogon/MultiPart.h>
#include <drogon/drogon.h>
#include <exception>
#include <string>

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

void run_ai_api()
{
    drogon::app().registerHandler(
        "/ai/run",
        [](const drogon::HttpRequestPtr&, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            callback(drogon::HttpResponse::newHttpJsonResponse(Json::Value(Json::objectValue)));
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
                        parser.getParameter<std::string>("tool"));
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
