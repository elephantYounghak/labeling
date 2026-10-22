#include "crud/project/model.h"

#include "utils/database.h"

void upload_project(Project& project)
{
    soci::session db = get_db();
    int id = 0;
    db << "INSERT INTO dbo.projects (name, description, path, created_at, updated_at) "
          "OUTPUT INSERTED.id "
          "VALUES (:name, :description, :path, CONVERT(DATETIME2, :created_at, 127), SYSUTCDATETIME())",
        soci::use(project.name, "name"),
        soci::use(project.description, "description"),
        soci::use(project.path, "path"),
        soci::use(project.created_at, "created_at"),
        soci::into(id);
    project.id = std::to_string(id);
}
