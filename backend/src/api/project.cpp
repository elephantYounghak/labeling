#include "api/project.h"

#include "models/project.h"

#include <drogon/drogon.h>
#include <exception>

void upload_project_api()
{
    drogon::app().registerHandler(
        "/project/upload",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            Project project;
            project.name = body ? (*body)["name"].asString() : "";
            project.description = body ? (*body)["description"].asString() : "";
            project.path = body ? (*body)["path"].asString() : "";
            project.created_at = body ? (*body)["createdAt"].asString() : "";
            project.updated_at = project.created_at;

            try {
                upload_project(project);
            } catch (const std::exception&) {
                Json::Value error;
                error["error"] = "프로젝트를 저장하지 못했습니다.";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }

            Json::Value saved;
            saved["id"] = project.id;
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Post});
}
