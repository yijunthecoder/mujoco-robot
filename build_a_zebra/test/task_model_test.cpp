#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "zebra_bt/task_model.hpp"

namespace {

const char * kTestBom = R"({
  "set_number": "99999",
  "role_order": ["legs", "body", "head"],
  "build_target": "zebra",
  "catalog": [
    {"part_id": "zebra_legs", "animal": "zebra", "role": "legs"},
    {"part_id": "zebra_body", "animal": "zebra", "role": "body"},
    {"part_id": "zebra_head", "animal": "zebra", "role": "head"},
    {"part_id": "tiger_legs", "animal": "tiger", "role": "legs"},
    {"part_id": "tiger_body", "animal": "tiger", "role": "body"},
    {"part_id": "tiger_head", "animal": "tiger", "role": "head"}
  ]
})";

class TaskModelTest : public ::testing::Test
{
protected:
  static std::string bom_path_;

  static void SetUpTestSuite()
  {
    bom_path_ = "/tmp/zebra_test_bom.json";
    std::ofstream f(bom_path_);
    f << kTestBom;
  }
};

std::string TaskModelTest::bom_path_;

TEST_F(TaskModelTest, LoadsBuildTarget)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_EQ(m.buildTarget(), "zebra");
}

TEST_F(TaskModelTest, BuildOrderSortedByRole)
{
  zebra_bt::TaskModel m(bom_path_);
  const auto & order = m.buildOrder();
  ASSERT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0].role, "legs");
  EXPECT_EQ(order[1].role, "body");
  EXPECT_EQ(order[2].role, "head");
}

TEST_F(TaskModelTest, BuildOrderContainsOnlyTargetAnimal)
{
  zebra_bt::TaskModel m(bom_path_);
  for (const auto & p : m.buildOrder()) {
    EXPECT_EQ(p.animal, "zebra");
  }
}

TEST_F(TaskModelTest, DependencyOfMiddlePart)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_EQ(m.dependencyOf("zebra_body"), "zebra_legs");
}

TEST_F(TaskModelTest, DependencyOfLastPart)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_EQ(m.dependencyOf("zebra_head"), "zebra_body");
}

TEST_F(TaskModelTest, FirstPartHasNoDependency)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_TRUE(m.dependencyOf("zebra_legs").empty());
}

TEST_F(TaskModelTest, UnknownPartHasNoDependency)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_TRUE(m.dependencyOf("not_a_real_part").empty());
}

TEST_F(TaskModelTest, FullCatalogIncludesAllAnimals)
{
  zebra_bt::TaskModel m(bom_path_);
  EXPECT_EQ(m.fullCatalog().size(), 6u);
}

TEST(TaskModelMissingFile, ThrowsRuntimeError)
{
  EXPECT_THROW(
    zebra_bt::TaskModel("/tmp/this_file_does_not_exist_zebra.json"),
    std::runtime_error);
}

}  // namespace