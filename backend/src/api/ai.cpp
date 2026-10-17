#include "api/ai.h"

#include <drogon/drogon.h>

void load_ai_api()
{
    drogon::app().registerHandler(
        "/ai/load",
        [](const drogon::HttpRequestPtr&, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            callback(drogon::HttpResponse::newHttpJsonResponse(Json::Value(Json::objectValue)));
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
