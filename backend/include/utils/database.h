#pragma once

#include <soci/soci.h>
#include <soci/odbc/soci-odbc.h>

inline soci::session get_db()
{
    return soci::session(
        soci::odbc,
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost\\SQLEXPRESS;"
        "Database=labeling;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;");
}
