#include "api/labeling.h"

#include "models/labeling.h"

#include <drogon/drogon.h>
#include <exception>

void upload_labeling_api()
{
    drogon::app().registerHandler(
        "/labeling/upload",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            Labeling labeling;
            labeling.project_id = body ? (*body)["projectId"].asString() : "";
            labeling.tool = body ? (*body)["tool"].asString() : "";
            labeling.opacity = body ? (*body)["opacity"].asDouble() : 0.4;
            labeling.classes = body ? (*body)["classes"].toStyledString() : "";

            try {
                upload_labeling(labeling);
            } catch (const std::exception&) {
                Json::Value error;
                error["error"] = "라벨링을 저장하지 못했습니다.";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }

            Json::Value saved;
            saved["projectId"] = labeling.project_id;
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Post});
}

void update_labeling_api()
{
    drogon::app().registerHandler(
        "/labeling/update",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            Labeling labeling;
            labeling.project_id = body ? (*body)["projectId"].asString() : "";
            labeling.tool = body ? (*body)["tool"].asString() : "";
            labeling.opacity = body ? (*body)["opacity"].asDouble() : 0.4;
            labeling.classes = body ? (*body)["classes"].toStyledString() : "";

            try {
                update_labeling(labeling);
            } catch (const std::exception&) {
                Json::Value error;
                error["error"] = "라벨링을 수정하지 못했습니다.";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }

            Json::Value saved;
            saved["projectId"] = labeling.project_id;
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Put});
}
