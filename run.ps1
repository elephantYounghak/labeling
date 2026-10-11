Set-Location $PSScriptRoot

$sqlServer = "localhost\SQLEXPRESS"
$sqlDatabase = "labeling"
$sqlFile = Join-Path $PSScriptRoot "database\tables.sql"

function Invoke-Mssql([string]$Database, [string]$Query) {
    $connection = New-Object System.Data.SqlClient.SqlConnection "Server=$sqlServer;Database=$Database;Integrated Security=True;"
    $connection.Open()
    try {
        $command = $connection.CreateCommand()
        $command.CommandText = $Query
        [void]$command.ExecuteNonQuery()
    }
    finally {
        $connection.Close()
    }
}

Invoke-Mssql "master" "IF DB_ID(N'$sqlDatabase') IS NULL CREATE DATABASE [$sqlDatabase];"
Invoke-Mssql $sqlDatabase (Get-Content -Raw -Path $sqlFile)

Start-Process powershell -WorkingDirectory "$PSScriptRoot\frontend" -ArgumentList "-NoExit", "-Command", "npm run dev"

Set-Location "$PSScriptRoot\backend"
xmake
xmake run
