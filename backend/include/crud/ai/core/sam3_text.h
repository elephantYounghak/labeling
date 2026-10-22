#pragma once

#include "crud/ai/core.h"

namespace sam3_text {

void load();
SamOutput detect(const SamInput& input);

}  // namespace sam3_text
