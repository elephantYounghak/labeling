add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})

add_requires("drogon", {
    configs = {
        mysql = false,
        redis = false,
        postgresql = false,
        sqlite3 = false,
    }
})
add_requires("opencv", {
    configs = {
        ffmpeg = false,
        cuda = false,
        opengl = false,
        vtk = false,
    }
})

add_requires("soci", {
    configs = {
        odbc = true,
    }
})

target("backend")
    set_kind("binary")
    set_languages("c++17")
    set_targetdir("bin")
    add_files("src/**.cpp")
    add_includedirs("include")
    add_linkdirs("lib")
    add_includedirs("lib")
    add_packages("drogon", "opencv", "soci")
    if is_plat("windows") then
        add_syslinks("ole32", "uuid", "shell32", "odbc32")
    end
