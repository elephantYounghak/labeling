#include "crud/ai/api.h"
#include "crud/io/api.h"
#include "crud/labeling/api.h"
#include "crud/project/api.h"
#include "utils/server.h"

#include <drogon/drogon.h>
#include <iostream>

int main()
{
    configure_drogon();
    upload_project_api();
    upload_labeling_api();
    update_labeling_api();
    upload_folder_api();
    load_ai_api();
    run_ai_api();
    detect_ai_api();
    drogon::app().addListener("0.0.0.0", 18080).run();
    return 0;
}
