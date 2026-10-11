#include "api/io.h"

#include <drogon/drogon.h>
#include <string>

void upload_folder_api()
{
    drogon::app().registerHandler(
        "/api/upload-folder",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            const auto body = req->getJsonObject();
            const std::string path = body ? (*body)["path"].asString() : "";

            Json::Value saved;
            saved["path"] = path;
            callback(drogon::HttpResponse::newHttpJsonResponse(saved));
        },
        {drogon::Post});
}
