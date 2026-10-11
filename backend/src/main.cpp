#include "api/io.h"
#include "api/labeling.h"
#include "api/project.h"
#include "api/api.h"

#include <drogon/drogon.h>
#include <iostream>

int main()
{
    configure_drogon();
    upload_project_api();
    upload_labeling_api();
    update_labeling_api();
    upload_folder_api();
    drogon::app().addListener("0.0.0.0", 18080).run();
    return 0;
}
