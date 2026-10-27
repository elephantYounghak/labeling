#include "crud/ai/core.h"
#include "crud/ai/core/sam3.h"
#include "crud/ai/core/sam3_point.h"
#include "crud/ai/core/sam3_text.h"

SamInput make_sam_input(std::string_view bytes, float x, float y, const std::string& tool, const std::string& word)
{
    return sam3::make_input(bytes, x, y, tool, word);
}

Json::Value sam_output_json(const SamOutput& output)
{
    return sam3::output_json(output);
}

void load_sam_text()
{
    sam3_text::load();
}

SamOutput detect_sam_text(const SamInput& input)
{
    return sam3_text::detect(input);
}

void load_sam_point()
{
    sam3_point::load();
}

SamOutput detect_sam_point(const SamInput& input)
{
    return sam3_point::detect(input);
}
