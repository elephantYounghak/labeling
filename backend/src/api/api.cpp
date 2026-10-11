#include "api/api.h"

#include <drogon/drogon.h>

void configure_drogon()
{
    drogon::app()
        .registerPreRoutingAdvice(
            [](const drogon::HttpRequestPtr& req,
               drogon::FilterCallback&& stop,
               drogon::FilterChainCallback&& pass) {
                if (req->path().rfind("/api/", 0) != 0 || req->method() != drogon::Options) {
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
            [](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
                if (req->path().rfind("/api/", 0) != 0) {
                    return;
                }
                resp->addHeader("Access-Control-Allow-Origin", "*");
            });
}
