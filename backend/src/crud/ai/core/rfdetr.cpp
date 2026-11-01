#include "crud/ai/core.h"

#include <filesystem>
#include <stdexcept>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

namespace fs = std::filesystem;


}  // namespace

std::string load_rfdetr()
{
    return "test";
}

std::string detect_rfdetr()
{
    return "test";
}

std::string make_train_command(const std::string& dataset_dir, const std::string& size,
    int epochs, int batch_size)
{
    const fs::path script = "backend/src/crud/ai/core/rfdetr.py";
    if (!fs::is_regular_file(script)) {
        throw std::runtime_error("학습 스크립트가 없습니다.");
    }

    std::string dataset = dataset_dir;
    if (!dataset.empty() && (dataset.back() == '\\' || dataset.back() == '/')) {
        dataset.pop_back();
    }

    return "uv run --project backend -- python -P \"" + script.generic_string() + "\" \"" + dataset + "\""
        + " --size " + size
        + " --epochs " + std::to_string(epochs)
        + " --batch-size " + std::to_string(batch_size);
}


void train_rfdetr(const std::string& dataset_dir, const std::string& size, 
    int epochs, int batch_size)
{
    if (dataset_dir.empty()) {
        throw std::runtime_error("학습 데이터 폴더가 없습니다.");
    }
    if (epochs < 1 || batch_size < 1) {
        throw std::runtime_error("epoch과 batch size는 1 이상이어야 합니다.");
    }

    const fs::path script = "backend/src/crud/ai/core/rfdetr.py";
    if (!fs::is_regular_file(script)) {
        throw std::runtime_error("학습 스크립트가 없습니다.");
    }

    std::string command = make_train_command(dataset_dir, size, epochs, batch_size);
    if (system(command.c_str()) != 0) {
        throw std::runtime_error("학습 스크립트가 실패했습니다.");
    }
}