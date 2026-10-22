#include "utils/server.h"

#include <drogon/drogon.h>

void configure_drogon()
{
    drogon::app()
        .setClientMaxBodySize(64 * 1024 * 1024)
        .setClientMaxMemoryBodySize(64 * 1024 * 1024)
        .registerPreRoutingAdvice(
            [](const drogon::HttpRequestPtr& req,
               drogon::FilterCallback&& stop,
               drogon::FilterChainCallback&& pass) {
                if (req->method() != drogon::Options) {
                    pass();
                    return;
                }

                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->addHeader("Access-Control-Allow-Origin", "*");
                resp->addHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
                resp->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
                stop(resp);
            })
        .registerPostHandlingAdvice(
            [](const drogon::HttpRequestPtr&, const drogon::HttpResponsePtr& resp) {
                resp->addHeader("Access-Control-Allow-Origin", "*");
            });
}
