add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})
set_policy("package.cmake_generator.ninja", false) -- ninja가 안 되는 문제가 종종 보고

package("cudnn")
    set_kind("library")
    set_homepage("https://developer.nvidia.com/cudnn")
    set_description("NVIDIA cuDNN 9.24 for CUDA 12.9")
    on_load(function (package)
        package:add("deps", "cuda 12.8.1", {
            system = false,
            configs = {
                utils = {"cudart", "cublas", "cufft", "nvrtc"},
            }
        })
    end)
    on_fetch(function (package, opt)
        local root = "C:/Program Files/NVIDIA/CUDNN/v9.24"
        local includedir = path.join(root, "include", "12.9")
        local linkdir = path.join(root, "lib", "12.9", "x64")
        local bindir = path.join(root, "bin", "12.9", "x64")
        if not os.isfile(path.join(linkdir, "cudnn.lib")) then
            return
        end
        return {
            version = "9.24",
            includedirs = includedir,
            linkdirs = linkdir,
            links = "cudnn",
            bindirs = bindir,
        }
    end)
    on_install("windows|x64", function (package) end)
package_end()

add_requires("drogon", {
    configs = {
        mysql = false,
        redis = false,
        postgresql = false,
        sqlite3 = false,
    }
})
add_requireconfs("**.trantor", {override = true, version = "v1.5.28"})
add_requires("soci", {
    configs = {
        odbc = true,
    }
})
add_requireconfs("**cuda", {
    override = true,
    system = false,
    version = "12.8.1",
    configs = {
        utils = {"cudart", "cublas", "cufft", "nvrtc"},
    }
})
add_requires("cuda 12.8.1", {
    system = false,
    configs = {
        utils = {"cudart", "cublas", "cufft", "nvrtc"},
    }
})
add_requires("cudnn 9.24", {system = true})
add_requires("onnxruntime 1.22.1", {
    configs = {
        gpu = true,
        cuda_version = "12",
    }
})
add_requires("opencv", {
    configs = {
        shared = true, -- 무슨 옵션인지 분석 
        ffmpeg = false,
        cuda = true,
        opengl = false,
        vtk = false,
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
    add_packages("drogon", "soci", "cuda", "cudnn", "onnxruntime", "opencv")
    add_runenvs("PATH", "C:/Program Files/NVIDIA/CUDNN/v9.24/bin/12.9/x64")
    if is_plat("windows") then
        add_syslinks("ole32", "uuid", "shell32", "odbc32")
    end
