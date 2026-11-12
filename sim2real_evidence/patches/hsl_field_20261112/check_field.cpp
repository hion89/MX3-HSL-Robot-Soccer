// Compile against the installed source header, not a copy of field constants.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "hsl_world_model/field_model.hpp"

void expect(bool value, const char * message) {
  if(!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
bool near(double a, double b) { return std::abs(a-b) < 1e-9; }
bool has(const std::vector<Eigen::Vector2d> & points, double x, double y) {
  for(const auto & p : points) if(near(p.x(), x) && near(p.y(), y)) return true;
  return false;
}
int main() {
  using hsl::FieldModel;
  using hsl::LandmarkType;
  const auto f = FieldModel::forProfile("practice");
  expect(near(f.length,3.9) && near(f.width,2.6), "line boundary");
  expect(near(f.center_circle_radius,.35), "circle radius");
  expect(near(f.penalty_mark_dist,.65), "penalty mark");
  expect(near(f.penalty_area_depth,.87) && near(f.penalty_area_width,2.17), "penalty area");
  expect(near(f.goal_area_depth,.43) && near(f.goal_area_width,1.30), "goal area");
  expect(near(f.goal_width,1.13), "goal opening");
  expect(near(f.length+2*f.border_x,5.0) && near(f.width+2*f.border_y,3.0), "carpet");
  expect(f.insideCarpet(Eigen::Vector2d(2.5,1.5)), "carpet edge included");
  expect(!f.insideCarpet(Eigen::Vector2d(2.501,0)), "outside long edge rejected");
  expect(!f.insideCarpet(Eigen::Vector2d(0,1.501)), "outside short edge rejected");
  expect(has(f.landmarks(LandmarkType::GOALPOST),-1.95,.565), "goal landmark");
  expect(has(f.landmarks(LandmarkType::T),-1.95,1.085), "penalty T");
  expect(has(f.landmarks(LandmarkType::T),-1.95,.65), "goal-area T");
  expect(has(f.landmarks(LandmarkType::L),-1.08,1.085), "penalty L");
  expect(has(f.landmarks(LandmarkType::L),-1.52,.65), "goal-area L");
  expect(has(f.landmarks(LandmarkType::X),-1.30,0), "penalty mark landmark");
  expect(has(f.landmarks(LandmarkType::X),0,.35), "circle landmark");
  const auto s = FieldModel::forProfile("s_field");
  expect(near(s.length,9) && near(s.width,6) && near(s.goal_width,1.8)
         && near(s.penalty_area_width,4) && near(s.center_circle_radius,.75), "S-Field unchanged");
  expect(s.insideCarpet(Eigen::Vector2d(5.5,4.0)), "S-Field carpet unchanged");
  std::cout << "PASS: C++ practice geometry/landmarks/carpet match photo; S-Field unchanged\n";
}
