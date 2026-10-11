CREATE TABLE #projects_keep (
    name NVARCHAR(200) NOT NULL,
    description NVARCHAR(1000) NOT NULL,
    path NVARCHAR(1024) NOT NULL,
    created_at DATETIME2 NOT NULL,
    updated_at DATETIME2 NOT NULL
);

IF EXISTS (
    SELECT 1
    FROM sys.columns
    WHERE object_id = OBJECT_ID(N'dbo.projects')
      AND name = N'id'
      AND system_type_id = TYPE_ID(N'uniqueidentifier')
)
BEGIN
    IF OBJECT_ID(N'dbo.labeling', N'U') IS NOT NULL
        DROP TABLE dbo.labeling;

    INSERT INTO #projects_keep (name, description, path, created_at, updated_at)
    SELECT name, description, path, created_at, updated_at
    FROM dbo.projects;

    DROP TABLE dbo.projects;
END

IF OBJECT_ID(N'dbo.projects', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.projects (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_projects PRIMARY KEY,
        name NVARCHAR(200) NOT NULL,
        description NVARCHAR(1000) NOT NULL CONSTRAINT DF_projects_description DEFAULT N'',
        path NVARCHAR(1024) NOT NULL,
        created_at DATETIME2 NOT NULL,
        updated_at DATETIME2 NOT NULL CONSTRAINT DF_projects_updated_at DEFAULT SYSUTCDATETIME()
    );
END

INSERT INTO dbo.projects (name, description, path, created_at, updated_at)
SELECT name, description, path, created_at, updated_at
FROM #projects_keep
ORDER BY created_at;

DROP TABLE #projects_keep;

IF OBJECT_ID(N'dbo.label_classes', N'U') IS NOT NULL
    DROP TABLE dbo.label_classes;

IF EXISTS (
    SELECT 1
    FROM sys.columns
    WHERE object_id = OBJECT_ID(N'dbo.labeling')
      AND name = N'project_id'
      AND system_type_id = TYPE_ID(N'uniqueidentifier')
)
    DROP TABLE dbo.labeling;

IF OBJECT_ID(N'dbo.labeling', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.labeling (
        project_id INT NOT NULL CONSTRAINT PK_labeling PRIMARY KEY,
        tool NVARCHAR(32) NOT NULL,
        opacity FLOAT NOT NULL CONSTRAINT DF_labeling_opacity DEFAULT 0.4,
        classes NVARCHAR(MAX) NOT NULL CONSTRAINT DF_labeling_classes DEFAULT N'[]',
        CONSTRAINT FK_labeling_projects FOREIGN KEY (project_id) REFERENCES dbo.projects(id)
    );
END
ELSE IF COL_LENGTH(N'dbo.labeling', N'classes') IS NULL
BEGIN
    ALTER TABLE dbo.labeling ADD classes NVARCHAR(MAX) NOT NULL CONSTRAINT DF_labeling_classes DEFAULT N'[]';
END
