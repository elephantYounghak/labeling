#include "crud/labeling/model.h"

#include "utils/database.h"

void upload_labeling(Labeling& labeling)
{
    soci::session db = get_db();
    db << "INSERT INTO dbo.labeling (project_id, tool, opacity, classes) "
          "VALUES (CAST(:project_id AS INT), :tool, :opacity, :classes)",
        soci::use(labeling.project_id, "project_id"),
        soci::use(labeling.tool, "tool"),
        soci::use(labeling.opacity, "opacity"),
        soci::use(labeling.classes, "classes");
}

void update_labeling(Labeling& labeling)
{
    soci::session db = get_db();
    db << "UPDATE dbo.labeling SET tool = :tool, opacity = :opacity, classes = :classes "
          "WHERE project_id = CAST(:project_id AS INT)",
        soci::use(labeling.tool, "tool"),
        soci::use(labeling.opacity, "opacity"),
        soci::use(labeling.classes, "classes"),
        soci::use(labeling.project_id, "project_id");
}
