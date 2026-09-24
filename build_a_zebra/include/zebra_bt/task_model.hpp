#pragma once                // Prevents the header from being included more than once during compilation.

#include <string>
#include <vector>
#include <stdexcept>

namespace zebra_bt
{

// One entry in the DUPLO parts catalog (a physical brick type).
// Template with 4 blank fields
struct PartSpec
{
  std::string part_id;      // e.g. "31111p0e" -- matches the real LDraw part file
  std::string animal;       // e.g. "zebra"
  std::string role;         // "legs" | "body" | "head"
  std::string description;  // human-readable, from the BOM file
};

// TaskModel is the static "recipe": the full parts catalog (all animals in
// the set) plus which animal we're building right now. It answers:
//   - which parts belong to the current build
//   - what order they must be assembled in (their dependencies)
//
// This is separate from WorldModel, which will track the *live* state of
// each part (located / picked / placed / lost) at runtime.
class TaskModel
{
// Public methods: exist purely so main.cpp is allowed to ask questions — "what's the build order?", "what does this part depend on?"
public:
  // Loads and parses a BOM json file like the one in resources/bom.json.
  // Throws std::runtime_error on a missing file or malformed JSON.
  explicit TaskModel(const std::string & bom_file_path);

  const std::vector<PartSpec> & fullCatalog() const {return catalog_;}
  const std::string & buildTarget() const {return build_target_;}

  // The parts belonging to the current build target, already ordered
  // bottom-to-top per role_order (e.g. legs -> body -> head).
  const std::vector<PartSpec> & buildOrder() const {return build_order_;}

  // Returns the part_id this part depends on (must be placed first), or
  // an empty string if it has no dependency (it's the first/base part).
  std::string dependencyOf(const std::string & part_id) const;

// Only can be change inside of this Class
private:
  std::vector<std::string> role_order_;
  std::string build_target_;
  std::vector<PartSpec> catalog_;
  std::vector<PartSpec> build_order_;
};

}  // namespace zebra_bt